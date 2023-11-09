# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from functools import wraps
from typing import Dict

import jax
from jax import core
from jax._src.util import safe_map

from ..core import ScaledArray

_scaled_ops_registry = {}


def register_scaled_op(lax_func, scaled_func):
    _scaled_ops_registry[lax_func] = scaled_func


def _get_lax_prim(scaled_func):
    try:
        op = getattr(jax.lax, scaled_func.__name__.replace("scaled_", ""))
    except AttributeError:
        raise AttributeError(f"Could not find corresponding jax.lax primitive for {scaled_func.__name__}")
    return op


def register_scaled_lax_op(scaled_func):
    """
    Registers a scaled function into the scaled_ops_registry by matching
    the function name with pattern `scaled_{func_name}` to a function in the
    `jax.lax` namespace.

    Example: `scaled_mul_p` is matched to `jax.lax.mul_p`
    """
    lax_prim = _get_lax_prim(scaled_func)
    register_scaled_op(lax_prim, scaled_func)


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


def autoscale_jaxpr(jaxpr, consts, *args):
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
        outvals = _scaled_ops_registry[eqn.primitive](*invals)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        safe_map(write, eqn.outvars, outvals)

    outvals = safe_map(read, jaxpr.outvars)
    if len(outvals) == 1:
        return outvals[0]
    else:
        return outvals
