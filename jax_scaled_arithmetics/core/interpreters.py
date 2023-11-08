# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from functools import wraps
from typing import Dict

import jax
from jax import core
from jax._src.util import safe_map

from ..core import ScaledArray

_scaled_ops_registry: Dict[callable, callable] = {}


def register_scaled_op(lax_func, scaled_func):
    _scaled_ops_registry[lax_func] = scaled_func


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
            outvals = [outvals]  # type: ignore
        safe_map(write, eqn.outvars, outvals)

    safe_map(read, jaxpr.outvars)

    return safe_map(read, jaxpr.outvars)
