# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax

from jax_scaled_arithmetics.core import debug_callback


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def debug_callback_grad(f, *args):
    """Custom callback, called on gradients."""
    return args


def debug_callback_grad_fwd(f, *args):
    return args, None


def debug_callback_grad_bwd(f, _, args_grad):
    debug_callback(f, *args_grad)
    return args_grad


debug_callback_grad.defvjp(debug_callback_grad_fwd, debug_callback_grad_bwd)


def debug_print(fmt: str, *args):
    """Debug print of a collection of tensors."""
    debug_callback(lambda *args: print(fmt.format(*args)), *args)
    return args


def debug_print_grad(fmt: str, *args):
    """Debug print of gradients of a collection of tensors."""
    return debug_callback_grad(lambda *args: print(fmt.format(*args)), *args)
