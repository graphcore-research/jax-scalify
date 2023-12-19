# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from enum import IntEnum
from functools import partial, wraps
from typing import Any, Dict, Sequence, Tuple

import jax
import numpy as np
from jax import core
from jax._src.custom_derivatives import (
    custom_jvp_call_jaxpr_p,
    custom_jvp_call_p,
    custom_vjp_call_jaxpr_p,
    custom_vjp_call_p,
)
from jax._src.util import safe_map

from .datatype import NDArray, ScaledArray, as_scaled_array_base, is_scaled_leaf
from .utils import Pow2RoundMode


@dataclass(frozen=True)
class AutoScaleConfig:
    """AutoScale configuration/parameters when tracing a graph.

    NOTE: this config can be locally changed using a Python context manager:
    `with AutoScaleConfig(...):`
    """

    rounding_mode: Pow2RoundMode = Pow2RoundMode.DOWN

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


def get_scale_dtype(val: Any) -> Any:
    if isinstance(val, ScaledArray):
        return val.scale.dtype
    return val.dtype


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


_scaled_ops_registry: Dict[core.Primitive, Tuple[Any, ScaledPrimitiveType]] = {}


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


def promote_scalar_to_scaled_array(val: Any) -> ScaledArray:
    """Promote a scalar (Numpy, JAX, ...) to a Scaled Array.

    Note: needs to work with any input type, including JAX tracer ones.
    """
    # Use `as_scaled_array` promotion rules.
    return as_scaled_array_base(val)


def numpy_constant_scaled_array(val: NDArray[Any]) -> ScaledArray:
    """Get the ScaledArray corresponding to a Numpy constant.

    Only supporting Numpy scalars at the moment.
    """
    # TODO: generalized rules!
    assert np.ndim(val) == 0
    assert np.issubdtype(val.dtype, np.floating)
    return ScaledArray(data=np.array(1.0, dtype=val.dtype), scale=np.copy(val))


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
        # Trace the graph & convert to scaled one.
        outputs_scaled_flat = autoscale_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *inputs_scaled_flat)
        # Reconstruct the output Pytree, with scaled arrays.
        # NOTE: this step is also handling single vs multi outputs.
        assert len(out_leaves) == len(outputs_scaled_flat)
        output_scaled = jax.tree_util.tree_unflatten(out_pytree, outputs_scaled_flat)
        return output_scaled

    return wrapped


def autoscale_jaxpr(jaxpr: core.Jaxpr, consts, *args):
    env: Dict[core.Var, ScaledArray] = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    def promote_to_scaled_array(val):
        if isinstance(val, ScaledArray):
            return val
        elif np.ndim(val) == 0:
            return promote_scalar_to_scaled_array(val)
        # No promotion rule => just return as such.
        return val

    # A few initial checks to make sure there is consistency.
    assert len(jaxpr.invars) == len(args)
    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        # Is there any ScaledArray among inputs?
        any_scaled_inputs = any([isinstance(v, ScaledArray) for v in invals])
        # Is there a scaled primitive associated?
        scaled_prim_fn, scaled_prim_type = _scaled_ops_registry.get(eqn.primitive, (None, ScaledPrimitiveType.NEVER))

        if not any_scaled_inputs and scaled_prim_type != ScaledPrimitiveType.ALWAYS_SCALE:
            # Using normal JAX primitive: no scaled inputs, and not always scale rule.
            subfuns, bind_params = eqn.primitive.get_bind_params(eqn.params)
            outvals = eqn.primitive.bind(*subfuns, *invals, **bind_params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
        elif scaled_prim_fn is None:
            raise NotImplementedError(
                f"'{eqn.primitive}' JAX primitive does not have an implementation for ScaledArray inputs yet."
            )
        else:
            # Using scaled primitive. Automatic promotion of inputs to scaled array, when possible.
            invals = list(map(promote_to_scaled_array, invals))
            outvals = scaled_prim_fn(*invals, **eqn.params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            print(
                "SCALED:",
                eqn.primitive,
                [(v.dtype, get_scale_dtype(v)) for v in invals],
                [(v.dtype, get_scale_dtype(v)) for v in outvals],
            )

        # if not eqn.primitive.multiple_results:
        #     outvals = [outvals]
        safe_map(write, eqn.outvars, outvals)

    outvals = safe_map(read, jaxpr.outvars)
    return outvals


def scaled_pjit_translation(*args: ScaledArray, **kwargs: Any) -> Sequence[ScaledArray]:
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

    # Generate the sub-scaled function, with proper `jax.jit` options.
    subfunc = partial(autoscale_jaxpr, closed_jaxpr.jaxpr, closed_jaxpr.literals)
    subfunc.__name__ = name  # type:ignore
    subfunc = jax.jit(subfunc, inline=inline, keep_unused=keep_unused)

    outputs_scaled_flat = subfunc(*args)
    return outputs_scaled_flat


try:
    from jax._src.pjit import pjit_p

    register_scaled_op(pjit_p, scaled_pjit_translation)
except (ImportError, ModuleNotFoundError):
    pass


def scaled_xla_call_translation(*args: ScaledArray, **kwargs: Any) -> Sequence[ScaledArray]:
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

    register_scaled_op(xla_call_p, scaled_xla_call_translation)
except (ImportError, ModuleNotFoundError):
    pass


def scaled_custom_jvp_call_translation(*args: ScaledArray, **params: Any) -> Sequence[ScaledArray]:
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


register_scaled_op(custom_jvp_call_p, scaled_custom_jvp_call_translation)
register_scaled_op(custom_jvp_call_jaxpr_p, scaled_custom_jvp_call_translation)


def scaled_custom_vjp_call_translation(*args: ScaledArray, **params: Any) -> Sequence[ScaledArray]:
    """Scaled translation of `custom_vjp_call` primitive. Forwarding the scaled call to sub-jaxpr,
    and modifying the underlying `vjp` function.
    """
    key_jaxpr = "fun_jaxpr"
    call_closed_jaxpr = params[key_jaxpr]
    # FIXME: re-call the custom_vjp decorator/bind.
    call_subfunc = partial(autoscale_jaxpr, call_closed_jaxpr.jaxpr, call_closed_jaxpr.literals)
    return call_subfunc(*args)


register_scaled_op(custom_vjp_call_p, scaled_custom_vjp_call_translation)
register_scaled_op(custom_vjp_call_jaxpr_p, scaled_custom_vjp_call_translation)
