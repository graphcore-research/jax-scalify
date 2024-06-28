# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
from functools import partial

import jax


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def map_on_forward(f, arg):
    """Map a function on a forward pass only. No-op/identity on backward pass."""
    return f(arg)


def map_on_forward_fwd(f, arg):
    return arg, None


def map_on_forward_bwd(f, _, grad):
    return (grad,)


map_on_forward.defvjp(map_on_forward_fwd, map_on_forward_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def map_on_backward(f, arg):
    """Map a function on the gradient/backward pass. No-op/identity on forward."""
    return arg


def map_on_backward_fwd(f, arg):
    return arg, None


def map_on_backward_bwd(f, _, grad):
    return (f(grad),)


map_on_backward.defvjp(map_on_backward_fwd, map_on_backward_bwd)
