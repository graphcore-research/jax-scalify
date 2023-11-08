# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import jax
import numpy as np

from jax import core
from jax._src.util import safe_map
from ..lax import scaled_ops_registry
from functools import wraps


def autoscale(fun):
    @wraps(fun)
    def wrapped(*args, **kwargs):
        unscaled_args = safe_map(lambda x: x.to_array() if hasattr(x, "to_array") else x, args)
        # get jaxpr of unscaled graph
        closed_jaxpr = jax.make_jaxpr(fun)(*unscaled_args, **kwargs)
        # convert to scaled graph
        out = autoscale_jaxpr(closed_jaxpr.jaxpr, closed_jaxpr.literals, *args)
        return out

    return wrapped


def autoscale_jaxpr(jaxpr, consts, *args):
    env = {}

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
        if eqn.primitive not in scaled_ops_registry:
            raise NotImplementedError(f"{eqn.primitive} does not have an implementation for ScaledArray inputs yet")
        outvals = scaled_ops_registry[eqn.primitive](*invals)
        if not eqn.primitive.multiple_results:
            outvals = [outvals]
        safe_map(write, eqn.outvars, outvals)

    safe_map(read, jaxpr.outvars)

    return safe_map(read, jaxpr.outvars)
