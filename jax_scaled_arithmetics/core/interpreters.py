# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from functools import wraps
from typing import Any, Dict

import jax
import numpy as np
from jax import core
from jax._src.util import safe_map

from .datatype import NDArray, ScaledArray

_scaled_ops_registry: Dict[core.Primitive, Any] = {}


def numpy_constant_scaled_array(val: NDArray[Any]) -> ScaledArray:
    """Get the ScaledArray corresponding to a Numpy constant.

    Only supporting Numpy scalars at the moment.
    """
    # TODO: generalized rules!
    assert val.shape == ()
    assert np.issubdtype(val.dtype, np.floating)
    return ScaledArray(data=np.array(1.0, dtype=val.dtype), scale=np.copy(val))


def register_scaled_op(prim: core.Primitive, scaled_func: Any) -> None:
    """Register the scaled translation of JAX primitive.

    Raises an error if a scaled translation is already existing for this primitive.

    Args:
        prim: JAX primitive.
        scaled_fund: Scaled translation of the primitive. With the same interface.
    """
    assert isinstance(prim, core.Primitive)
    if prim in _scaled_ops_registry:
        raise KeyError(f"A scaled translation is already registered for the JAX primitive '{prim}'.")
    _scaled_ops_registry[prim] = scaled_func


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


def register_scaled_lax_op(scaled_func):
    """
    Registers a scaled function/translation into the scaled_ops_registry by matching
    the function name with pattern `scaled_{func_name}` to a primitive in the
    `jax.lax` namespace.

    Example: `scaled_mul` is matched to `jax.lax.mul_p`
    """
    lax_prim = _get_lax_prim(scaled_func)
    register_scaled_op(lax_prim, scaled_func)
    # Always return the function in the case of decorator use.
    return scaled_func


def autoscale(fun):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        aval_args = safe_map(lambda x: x.aval, args)
        # Get jaxpr of unscaled/normal graph. Getting output Pytree shape as well.
        closed_jaxpr, outshape = jax.make_jaxpr(fun, return_shape=True)(*aval_args, **kwargs)
        out_leaves, out_pytree = jax.tree_util.tree_flatten(outshape)
        # Trace the graph & convert to scaled one.
        outputs_flat = autoscale_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        # Reconstruct the output Pytree, with scaled arrays.
        # NOTE: this step is also handling single vs multi outputs.
        assert len(out_leaves) == len(outputs_flat)
        output = jax.tree_util.tree_unflatten(out_pytree, outputs_flat)
        return output

    return wrapped


def autoscale_jaxpr(jaxpr: core.Jaxpr, consts, *args):
    env: Dict[core.Var, ScaledArray] = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    def to_scaled_array(val):
        if isinstance(val, ScaledArray):
            return val
        elif isinstance(val, np.ndarray):
            return numpy_constant_scaled_array(val)
        raise TypeError(f"Can not convert '{val}' to a scaled array.")

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        # Make sure all inputs are scaled arrays
        invals = list(map(to_scaled_array, invals))
        assert all([isinstance(v, ScaledArray) for v in invals])
        # TODO: handle `stop_scale` case? integer/boolean dtypes?

        # Primitive is supported by `autoscale`?
        if eqn.primitive not in _scaled_ops_registry:
            raise NotImplementedError(
                f"'{eqn.primitive}' JAX primitive does not have an implementation for ScaledArray inputs yet."
            )
        outvals = _scaled_ops_registry[eqn.primitive](*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        safe_map(write, eqn.outvars, outvals)

    outvals = safe_map(read, jaxpr.outvars)
    return outvals
