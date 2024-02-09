# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from enum import IntEnum
from functools import partial, wraps
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union

import jax
import numpy as np
from jax import core, lax
from jax._src.custom_derivatives import (
    custom_jvp_call_jaxpr_p,
    custom_jvp_call_p,
    custom_vjp_call_jaxpr_p,
    custom_vjp_call_p,
)
from jax._src.util import safe_map
from jax.tree_util import register_pytree_node_class

from .datatype import (
    Array,
    ArrayTypes,
    DTypeLike,
    ScaledArray,
    Shape,
    as_scaled_array_base,
    is_scaled_leaf,
    is_static_zero,
)
from .pow2 import Pow2RoundMode
from .utils import python_scalar_as_numpy


@dataclass(frozen=True)
class AutoScaleConfig:
    """AutoScale configuration/parameters when tracing a graph.

    NOTE: this config can be locally changed using a Python context manager:
    `with AutoScaleConfig(...):`

    Args:
        rounding_mode: Power-of-2 rounding mode.
        scale_dtype: Scale (default) datatype.
    """

    rounding_mode: Pow2RoundMode = Pow2RoundMode.DOWN
    scale_dtype: DTypeLike = None

    def __enter__(self):
        global _autoscale_config_stack
        _autoscale_config_stack.append(self)

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _autoscale_config_stack
        _autoscale_config_stack.pop()


# AutoScale config stack.
_autoscale_config_stack = [AutoScaleConfig()]


def get_autoscale_config() -> AutoScaleConfig:
    """Get current/local autoscale config."""
    return _autoscale_config_stack[-1]


class ScaledPrimitiveType(IntEnum):
    """Scale (JAX) primitive type.

    This enum described the behaviour when `autoscale` is
    tracing the graph.

    FORWARD: Forwarding scaling => only used if scaled inputs.
        Default behaviour.
    ALWAYS_SCALE: Always use scaled version.
    """

    NEVER = 0
    FORWARD = 1
    ALWAYS_SCALE = 2


_scaled_jaxpr_ops_registry: Dict[core.Primitive, Any] = {}
"""Registry of (sub) "jaxpr" ops/primitives and their scaled translation.

The core "jaxpr" primitives are typical `pjit`, `xla_call`, where the JSA interpreter
will need to be run on sub-jaxprs, passing the full metadata on input/output tensors.
"""


_scaled_ops_registry: Dict[core.Primitive, Tuple[Any, ScaledPrimitiveType]] = {}
"""Registry of JAX common primitives and their scaled translation.
"""


_broadcasted_scalar_preserving_primitives: Set[core.Primitive] = set()
"""Scalar (broadcasted array) preserving JAX primitives

More specifically: if all inputs are (broadcasted) scalars, then the output(s)
are broadcasted scalars. Keeping track of broadcasted scalars is allowing
proper conversion to ScaledArrays (instead of assigning default scale 1).
"""

_broadcasted_zero_preserving_primitives: Set[core.Primitive] = set()
"""Zero (broadcasted array) preserving JAX primitives

More specifically: if all inputs are (broadcasted) zero scalars, then the output(s)
are broadcasted zero scalars. Keeping track of broadcasted zero scalars is allowing
proper conversion to ScaledArrays (instead of assigning default scale 1).
"""


def _get_lax_prim(scaled_func: Any) -> core.Primitive:
    try:
        prim_name = scaled_func.__name__.replace("scaled_", "") + "_p"
        prim = getattr(jax.lax, prim_name)
    except AttributeError:
        raise AttributeError(f"Could not find corresponding 'jax.lax' primitive for '{scaled_func.__name__}'.")
    # Check as well it is a proper primitive! And not something else also in `jax.lax`
    if not isinstance(prim, core.Primitive):
        raise AttributeError(f"The object `{prim}` is not a proper JAX primitive for '{scaled_func.__name__}'.")
    return prim


def _get_aval(val: Any) -> core.ShapedArray:
    """Get the abstract value (i.e. ShapedArray) from any input."""
    if hasattr(val, "aval"):
        return val.aval
    return core.ShapedArray(shape=val.shape, dtype=val.dtype)


def _get_data(val: Any) -> Array:
    if isinstance(val, ScaledArray):
        return val.data
    return val


def register_scaled_op(
    prim: core.Primitive, scaled_func: Any, scaled_type: ScaledPrimitiveType = ScaledPrimitiveType.FORWARD
) -> None:
    """Register the scaled translation of JAX primitive.

    Raises an error if a scaled translation is already existing for this primitive.

    Args:
        prim: JAX primitive.
        scaled_func: Scaled translation of the primitive. With the same interface.
        scaled_type: Scaled primitive type => behaviour when `autoscale` tracing.
    """
    assert isinstance(prim, core.Primitive)
    # Can not register a jaxpr type op this way.
    assert prim not in _scaled_jaxpr_ops_registry
    if prim in _scaled_ops_registry:
        raise KeyError(f"A scaled translation is already registered for the JAX primitive '{prim}'.")
    _scaled_ops_registry[prim] = (scaled_func, scaled_type)


def register_scaled_lax_op(scaled_func):
    """
    Registers a scaled function/translation into the scaled_ops_registry by matching
    the function name with pattern `scaled_{func_name}` to a primitive in the
    `jax.lax` namespace.

    Example: `scaled_mul` is matched to `jax.lax.mul_p`
    """
    lax_prim = _get_lax_prim(scaled_func)
    register_scaled_op(lax_prim, scaled_func, ScaledPrimitiveType.FORWARD)
    # Always return the function in the case of decorator use.
    return scaled_func


def find_registered_scaled_op(prim: core.Primitive) -> Tuple[Any, ScaledPrimitiveType]:
    """Find a registered JAX scaled operation/translation. Returns (None, None) if
    the primitive does not have a scaled translation registered.

    Args:
        prim: JAX primitive.
    """
    return _scaled_ops_registry.get(prim, (None, ScaledPrimitiveType.NEVER))


@register_pytree_node_class
@dataclass(frozen=True, init=False)
class ScalifyTracerArray:
    """Meta-Array class used in scalify tracer. It can represent
    any array, scaled or not, and tracks whether an array corresponds to a scalar broadcasted.

    Compatible with JAX PyTrees in order to be able to trace a graph with `ScalifyTracerArray`
    as inputs/outputs.

    Args:
        array: Normal or scaled array.
        is_broadcasted_scalar: Is the array a broadcasted scalar (metadata).
        is_broadcasted_zero: Is the array a broadcased zero scalar (metadata).
    """

    array: Union[Array, ScaledArray] = None
    # Metadata fields.
    is_broadcasted_scalar: bool = False
    is_broadcasted_zero: bool = False

    def __init__(
        self,
        arr: Union[Array, ScaledArray],
        is_broadcasted_scalar: Optional[bool] = None,
        is_broadcasted_zero: Optional[bool] = None,
    ) -> None:
        # Convert Python scalars, if necessary.
        arr = python_scalar_as_numpy(arr)
        assert isinstance(arr, (np.bool_, np.number, np.ndarray, ScaledArray, *ArrayTypes))
        object.__setattr__(self, "array", arr)

        # Is a zero broadcasted scalar? Only checking when info is not provided.
        if is_broadcasted_zero is None:
            is_broadcasted_zero = bool(np.all(is_static_zero(self.array)))
        object.__setattr__(self, "is_broadcasted_zero", is_broadcasted_zero)
        # Optional is broadcasted scalar information (always consistent with broadcasted zero!)
        is_scalar = self.array.size == 1
        is_broadcasted_scalar = is_scalar if is_broadcasted_scalar is None else is_broadcasted_scalar or is_scalar
        is_broadcasted_scalar = is_broadcasted_scalar or is_broadcasted_zero
        object.__setattr__(self, "is_broadcasted_scalar", is_broadcasted_scalar)

        # Always make sure we have zero scale to represent broadcasted zero.
        if is_broadcasted_zero and isinstance(self.array, ScaledArray):
            object.__setattr__(self.array, "scale", np.array(0, self.array.scale.dtype))

    def tree_flatten(self):
        # See official JAX documentation on extending PyTrees.
        # Note: using explicit tree flatten instead of chex for MyPy compatibility.
        children = (self.array,)
        aux_data = (self.is_broadcasted_scalar, self.is_broadcasted_zero)
        return (children, aux_data)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # See official JAX documentation on extending PyTrees.
        assert len(aux_data) == 2
        assert len(children) == 1
        return cls(children[0], is_broadcasted_scalar=aux_data[0], is_broadcasted_zero=aux_data[1])

    @property
    def dtype(self) -> DTypeLike:
        return self.array.dtype

    @property
    def size(self) -> int:
        return self.array.size

    @property
    def shape(self) -> Shape:
        return self.array.shape

    @property
    def is_scaled_array(self) -> bool:
        return isinstance(self.array, ScaledArray)

    def to_scaled_array(self, scale_dtype: Optional[DTypeLike] = None) -> ScaledArray:
        """(Tentatively) converting to a scaled array.

        Supporting the following cases:
            - scalar array;
            - broadcasted scalar array;

        Not supporting:
            - bool/int dtypes;
            - any other array;

        TODO: support (constant) Numpy arrays.
        """
        # Already scaled array, or not a floating point dtype.
        if isinstance(self.array, ScaledArray) or not np.issubdtype(self.dtype, np.floating):
            return self.array

        if self.is_broadcasted_zero:
            # Directly create the scaled array.
            scale_dtype = scale_dtype or self.array.dtype
            scale = np.array(0, dtype=scale_dtype)
            return ScaledArray(data=self.array, scale=scale)
        elif np.ndim(self.array) == 0:
            # Single value => "easy case".
            return as_scaled_array_base(self.array, scale_dtype=scale_dtype)
        elif self.is_broadcasted_scalar:
            # Broadcasted scalar => convert as a scalar.
            scalar_val = self.array.ravel()[0]
            scaled_scalar = as_scaled_array_base(scalar_val, scale_dtype=scale_dtype)
            return as_scaled_array_base(self.array, scale=scaled_scalar.scale)

        # No promotion rule found => just return as such.
        return self.array

    def to_array(self) -> Array:
        """Converting to a (normal) JAX/Numpy array."""
        if not self.is_scaled_array:
            return self.array
        return self.array.to_array()


def autoscale(fun):
    """`autoscale` JAX graph transformation.

    The `autoscale` graph transformation works in a forwarding mode:
        scaled arrays are forwarded to scaled primitives, which will generate scaled outputs.

    If no inputs to a JAX primitive are scaled -> the normal primitive is then called, generating a common
    JAX output array.

    This behaviour is the standard one for `ScaledPrimitiveType.FORWARD` primitives.
    An alternative behaviour is possible for `ScaledPrimitiveType.ALWAYS_SCALED` primitives, where the scaled
    operation will always be called. A typical example is the `set_scaling` primitive.
    """

    @wraps(fun)
    def wrapped(*args, **kwargs):
        if len(kwargs) > 0:
            raise NotImplementedError("`autoscale` JAX interpreter not supporting named tensors at present.")

        aval_args = jax.tree_map(_get_aval, args, is_leaf=is_scaled_leaf)
        # Get jaxpr of unscaled/normal graph. Getting output Pytree shape as well.
        closed_jaxpr, outshape = jax.make_jaxpr(fun, return_shape=True)(*aval_args, **kwargs)
        out_leaves, out_pytree = jax.tree_util.tree_flatten(outshape)

        # Flattening of PyTree inputs.
        inputs_scaled = args
        inputs_scaled_flat, _ = jax.tree_util.tree_flatten(inputs_scaled, is_leaf=is_scaled_leaf)
        # Convert to Scalify tracer (meta) arrays.
        inputs_tracer_flat = list(map(ScalifyTracerArray, inputs_scaled_flat))
        consts_tracer_flat = list(map(ScalifyTracerArray, closed_jaxpr.literals))
        # Trace the graph & convert to scaled one.
        outputs_tracer_flat = autoscale_jaxpr(closed_jaxpr.jaxpr, consts_tracer_flat, *inputs_tracer_flat)
        outputs_scaled_flat = [v.array for v in outputs_tracer_flat]
        # Reconstruct the output Pytree, with scaled arrays.
        # NOTE: this step is also handling single vs multi outputs.
        assert len(out_leaves) == len(outputs_scaled_flat)
        output_scaled = jax.tree_util.tree_unflatten(out_pytree, outputs_scaled_flat)
        return output_scaled

    return wrapped


def jaxpr_eqn_bind(eqn: core.JaxprEqn, invals: Sequence[core.ShapedArray]) -> Sequence[core.ShapedArray]:
    """Bind a Jaxpr equation to arrays."""
    subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
    outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
    if not eqn.primitive.multiple_results:
        outvals = [outvals]
    return outvals


def autoscale_jaxpr(
    jaxpr: core.Jaxpr, consts: Sequence[ScalifyTracerArray], *args: ScalifyTracerArray
) -> Sequence[ScalifyTracerArray]:
    env: Dict[core.Var, ScalifyTracerArray] = {}
    # Check dtype consistency between normal and scaled modes.
    safe_check_dtypes: bool = False
    # AutoScale config to use.
    autoscale_cfg = get_autoscale_config()

    def read(var: core.Var) -> ScalifyTracerArray:
        if type(var) is core.Literal:
            # Wrap the constant in tracer array.
            return ScalifyTracerArray(var.val)
        return env[var]

    def write(var: core.Var, val: ScalifyTracerArray) -> None:
        env[var] = val

    # A few initial checks to make sure there is consistency.
    assert len(jaxpr.invars) == len(args)
    assert len(jaxpr.constvars) == len(consts)
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals_tracer: List[ScalifyTracerArray] = safe_map(read, eqn.invars)
        if eqn.primitive in _scaled_jaxpr_ops_registry:
            # Core sub-jaxpr primitive => pass the complete tracer array with metadata.
            scaled_jaxpr_prim_fn = _scaled_jaxpr_ops_registry[eqn.primitive]
            outvals_tracer = scaled_jaxpr_prim_fn(*invals_tracer, **eqn.params)
            # Save outputs and move on!
            safe_map(write, eqn.outvars, outvals_tracer)
            continue

        # Common (scaled) JAX primitives path.
        # Is there any ScaledArray among inputs?
        any_scaled_inputs = any([v.is_scaled_array for v in invals_tracer])
        # Is there a scaled primitive associated?
        scaled_prim_fn, scaled_prim_type = _scaled_ops_registry.get(eqn.primitive, (None, ScaledPrimitiveType.NEVER))

        # Are outputs broadcasted scalars?
        are_inputs_broadcasted_scalars = all([v.is_broadcasted_scalar for v in invals_tracer])
        are_outputs_broadcasted_scalars = (
            are_inputs_broadcasted_scalars and eqn.primitive in _broadcasted_scalar_preserving_primitives
        )
        # Are outputs broadcasted zeroes?
        are_inputs_broadcasted_zeroes = all([v.is_broadcasted_zero for v in invals_tracer])
        are_outputs_broadcasted_zeroes = (
            are_inputs_broadcasted_zeroes and eqn.primitive in _broadcasted_zero_preserving_primitives
        )
        # Outputs scalify factory method.
        scalify_array_init_fn = lambda v: ScalifyTracerArray(
            v, is_broadcasted_scalar=are_outputs_broadcasted_scalars, is_broadcasted_zero=are_outputs_broadcasted_zeroes
        )

        if not any_scaled_inputs and scaled_prim_type != ScaledPrimitiveType.ALWAYS_SCALE:
            # Using normal JAX primitive: no scaled inputs, and not always scale rule.
            invals = [v.to_array() for v in invals_tracer]
            outvals = jaxpr_eqn_bind(eqn, invals)
            outvals_tracer = list(map(scalify_array_init_fn, outvals))
        elif scaled_prim_fn is None:
            raise NotImplementedError(
                f"'{eqn.primitive}' JAX primitive does not have an implementation for ScaledArray inputs yet."
            )
        else:
            # Using scaled primitive. Automatic promotion of inputs to scaled array, when possible.
            scaled_invals = [v.to_scaled_array(autoscale_cfg.scale_dtype) for v in invals_tracer]
            outvals = scaled_prim_fn(*scaled_invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            outvals_tracer = list(map(scalify_array_init_fn, outvals))

            # Check consistency with normal JAX mode. Help catching dtype promotion errors.
            # NOTE: ignoring when no outputs! (e.g. debug_callback).
            if safe_check_dtypes and len(outvals) > 0:
                ref_outvals = jaxpr_eqn_bind(eqn, [_get_data(v.array) for v in invals_tracer])
                data_outvals = [_get_data(v) for v in outvals]
                # Check scaled dtypes == ref dtypes.
                ref_dtypes = tuple(v.dtype for v in ref_outvals)
                data_dtypes = tuple(v.dtype for v in data_outvals)
                if data_dtypes != ref_dtypes:
                    raise ValueError(
                        f"Output dtype of '{eqn.primitive}' scaled translation is not consistent with the JAX reference primitive implementation: {data_dtypes} vs {ref_dtypes}."
                    )

        safe_map(write, eqn.outvars, outvals_tracer)

    outvals_tracer = safe_map(read, jaxpr.outvars)
    return outvals_tracer


def scaled_pjit_translation(*args: ScalifyTracerArray, **kwargs: Any) -> Sequence[ScalifyTracerArray]:
    """Scaled translation of `pjit`. Basically re-running `autoscale` on sub-jaxpr.

    NOTE: the `pjit` call will be kept, forwarding the proper parameters (shardings, ...).
    """
    closed_jaxpr = kwargs["jaxpr"]
    name = kwargs["name"]
    inline = kwargs["inline"]
    keep_unused = kwargs["keep_unused"]
    # TODO: properly adapt + pass these options.
    # donated_invars = kwargs["donated_invars"]
    # in_shardings = kwargs["in_shardings"]
    # out_shardings = kwargs["out_shardings"]

    consts_tracer_flat = [ScalifyTracerArray(v) for v in closed_jaxpr.literals]
    # Generate the sub-scaled function, with proper `jax.jit` options.
    subfunc = partial(autoscale_jaxpr, closed_jaxpr.jaxpr, consts_tracer_flat)
    subfunc.__name__ = name  # type:ignore
    subfunc = jax.jit(subfunc, inline=inline, keep_unused=keep_unused)
    outvals = subfunc(*args)
    return outvals


try:
    from jax._src.pjit import pjit_p

    _scaled_jaxpr_ops_registry[pjit_p] = scaled_pjit_translation
except (ImportError, ModuleNotFoundError):
    pass


def scaled_xla_call_translation(*args: ScalifyTracerArray, **kwargs: Any) -> Sequence[ScalifyTracerArray]:
    """Scaled translation of `xla_call`. Basically re-running `autoscale` on sub-jaxpr.

    Useful for JAX 0.3 compatibility
    """
    jaxpr = kwargs["call_jaxpr"]
    name = kwargs["name"]
    inline = kwargs["inline"]
    keep_unused = kwargs["keep_unused"]
    # TODO: properly adapt + pass these options.
    # donated_invars = kwargs["donated_invars"]
    # in_shardings = kwargs["in_shardings"]
    # out_shardings = kwargs["out_shardings"]

    assert len(jaxpr.constvars) == 0
    # Generate the sub-scaled function, with proper `jax.jit` options.
    subfunc = partial(autoscale_jaxpr, jaxpr, [])
    subfunc.__name__ = name  # type:ignore
    subfunc = jax.jit(subfunc, inline=inline, keep_unused=keep_unused)
    outputs_scaled_flat = subfunc(*args)
    return outputs_scaled_flat


try:
    from jax.interpreters.xla import xla_call_p

    _scaled_jaxpr_ops_registry[xla_call_p] = scaled_xla_call_translation
except (ImportError, ModuleNotFoundError):
    pass


def scaled_custom_jvp_call_translation(*args: ScalifyTracerArray, **params: Any) -> Sequence[ScalifyTracerArray]:
    """Scaled translation of `custom_jvp_call` primitive. Forwarding the scaled call to sub-jaxpr,
    and modifying the underlying `jvp` function.
    """
    # [fun, jvp], bind_params = custom_jvp_call_p.get_bind_params(params)
    key_jaxpr = "call_jaxpr" if jax.__version_info__[1] > 3 else "fun_jaxpr"
    call_closed_jaxpr = params[key_jaxpr]
    # JAX 0.3 compatibility.
    assert params.get("num_consts", 0) == 0
    # FIXME: re-call the custom_jvp decorator/bind.
    call_subfunc = partial(autoscale_jaxpr, call_closed_jaxpr.jaxpr, call_closed_jaxpr.literals)
    return call_subfunc(*args)


_scaled_jaxpr_ops_registry[custom_jvp_call_p] = scaled_custom_jvp_call_translation
_scaled_jaxpr_ops_registry[custom_jvp_call_jaxpr_p] = scaled_custom_jvp_call_translation


def scaled_custom_vjp_call_translation(*args: ScalifyTracerArray, **params: Any) -> Sequence[ScalifyTracerArray]:
    """Scaled translation of `custom_vjp_call` primitive. Forwarding the scaled call to sub-jaxpr,
    and modifying the underlying `vjp` function.
    """
    key_jaxpr = "fun_jaxpr"
    call_closed_jaxpr = params[key_jaxpr]
    # FIXME: re-call the custom_vjp decorator/bind.
    call_subfunc = partial(autoscale_jaxpr, call_closed_jaxpr.jaxpr, call_closed_jaxpr.literals)
    return call_subfunc(*args)


_scaled_jaxpr_ops_registry[custom_vjp_call_p] = scaled_custom_vjp_call_translation
_scaled_jaxpr_ops_registry[custom_vjp_call_jaxpr_p] = scaled_custom_vjp_call_translation


# Default collection of scalar preserving JAX primitives.
_broadcasted_scalar_preserving_primitives |= {
    lax.abs_p,
    lax.acos_p,
    lax.acosh_p,
    lax.add_p,
    lax.asin_p,
    lax.asinh_p,
    lax.atan_p,
    lax.atan2_p,
    lax.atanh_p,
    lax.bitcast_convert_type_p,
    lax.broadcast_in_dim_p,
    lax.cbrt_p,
    lax.ceil_p,
    lax.clamp_p,
    lax.convert_element_type_p,
    lax.exp_p,
    lax.expm1_p,
    lax.floor_p,
    lax.integer_pow_p,
    lax.min_p,
    lax.max_p,
    lax.mul_p,
    lax.neg_p,
    lax.reduce_prod_p,
    lax.reduce_sum_p,
    lax.reduce_max_p,
    lax.reduce_min_p,
    lax.reduce_precision_p,
    lax.reshape_p,
    lax.rem_p,
    lax.slice_p,
    lax.sin_p,
    lax.sinh_p,
    lax.sub_p,
    lax.sqrt_p,
    lax.tan_p,
    lax.tanh_p,
    lax.transpose_p,
}

# Default collection of zero arrays preserving JAX primitives.
_broadcasted_zero_preserving_primitives |= {
    lax.abs_p,
    lax.add_p,
    lax.broadcast_in_dim_p,
    lax.cbrt_p,
    lax.ceil_p,
    lax.clamp_p,
    lax.convert_element_type_p,
    lax.exp_p,
    lax.expm1_p,
    lax.floor_p,
    lax.min_p,
    lax.max_p,
    lax.mul_p,
    lax.neg_p,
    lax.reduce_prod_p,
    lax.reduce_sum_p,
    lax.reduce_max_p,
    lax.reduce_min_p,
    lax.reduce_precision_p,
    lax.reshape_p,
    lax.slice_p,
    lax.sin_p,
    lax.sinh_p,
    lax.sub_p,
    lax.sqrt_p,
    lax.tan_p,
    lax.tanh_p,
    lax.transpose_p,
}
