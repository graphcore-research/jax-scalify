# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from functools import wraps
from typing import Any, Dict

import jax
from jax import core
from jax._src.util import safe_map

from ..core import ScaledArray

_scaled_ops_registry: Dict[core.Primitive, Any] = {}


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
        # get jaxpr of unscaled graph
        closed_jaxpr = jax.make_jaxpr(fun)(*aval_args, **kwargs)
        # convert to scaled graph
        out = autoscale_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out

    return wrapped


def autoscale_jaxpr(jaxpr: core.Jaxpr, consts, *args):
    env: Dict[core.Var, ScaledArray] = {}

    def read(var):
        if type(var) is core.Literal:
            return var.val
        return env[var]

    def write(var, val):
        env[var] = val

    safe_map(write, jaxpr.invars, args)
    safe_map(write, jaxpr.constvars, consts)

    for eqn in jaxpr.eqns:
        invals = safe_map(read, eqn.invars)
        if eqn.primitive not in _scaled_ops_registry:
            raise NotImplementedError(f"{eqn.primitive} does not have an implementation for ScaledArray inputs yet")
        outvals = _scaled_ops_registry[eqn.primitive](*invals, **eqn.params)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        safe_map(write, eqn.outvars, outvals)

    outvals = safe_map(read, jaxpr.outvars)
    if len(outvals) == 1:
        return outvals[0]
    else:
        return outvals
