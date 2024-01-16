# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax
import numpy as np

from jax_scaled_arithmetics.core import ScaledArray, pow2_round, pow2_round_down
from jax_scaled_arithmetics.lax import get_data_scale, rebalance


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fn_fwd_identity_bwd(f, arg):
    """Function with identity bwd/grad."""
    return f(arg)


def fn_fwd_identity_bwd_fwd(f, arg):
    return arg, None


def fn_fwd_identity_bwd_bwd(f, _, grad):
    return (grad,)


fn_fwd_identity_bwd.defvjp(fn_fwd_identity_bwd_fwd, fn_fwd_identity_bwd_bwd)


@partial(jax.custom_vjp, nondiff_argnums=(0,))
def fn_bwd_identity_fwd(f, arg):
    """Apply a function on the gradient/backward pass."""
    return arg


def fn_bwd_identity_fwd_fwd(f, arg):
    return arg, None


def fn_bwd_identity_fwd_bwd(f, _, grad):
    return (f(grad),)


fn_bwd_identity_fwd.defvjp(fn_bwd_identity_fwd_fwd, fn_bwd_identity_fwd_bwd)


def dynamic_rescale_max_base(arr: ScaledArray) -> ScaledArray:
    """Dynamic rescaling of a ScaledArray, using abs-max."""
    # Similarly to ML norms => need some epsilon for training stability!
    eps = pow2_round_down(np.float32(1e-4))

    data, scale = get_data_scale(arr)
    data_sq = jax.lax.abs(data)
    axes = tuple(range(data.ndim))
    # Get MAX norm + pow2 rounding.
    norm = jax.lax.reduce_max_p.bind(data_sq, axes=axes)
    norm = jax.lax.max(pow2_round(norm).astype(scale.dtype), eps.astype(scale.dtype))
    # Rebalancing based on norm.
    return rebalance(arr, norm)


def dynamic_rescale_l1_base(arr: ScaledArray) -> ScaledArray:
    """Dynamic rescaling of a ScaledArray, using L1 norm.

    NOTE: by default, computing L1 norm in FP32.
    """
    # Similarly to ML norms => need some epsilon for training stability!
    norm_dtype = np.float32
    eps = pow2_round_down(norm_dtype(1e-4))

    data, scale = get_data_scale(arr)
    data_sq = jax.lax.abs(data.astype(np.float32))
    axes = tuple(range(data.ndim))
    # Get L1 norm + pow2 rounding.
    norm = jax.lax.reduce_sum_p.bind(data_sq, axes=axes) / data.size
    norm = jax.lax.max(pow2_round(norm), eps).astype(scale.dtype)
    # Rebalancing based on norm.
    return rebalance(arr, norm)


def dynamic_rescale_l2_base(arr: ScaledArray) -> ScaledArray:
    """Dynamic rescaling of a ScaledArray, using L2 norm.

    NOTE: by default, computing L2 norm in FP32.
    """
    # Similarly to ML norms => need some epsilon for training stability!
    norm_dtype = np.float32
    eps = pow2_round_down(norm_dtype(1e-4))

    data, scale = get_data_scale(arr)
    data_sq = jax.lax.integer_pow(data.astype(norm_dtype), 2)
    axes = tuple(range(data.ndim))
    # Get L2 norm + pow2 rounding.
    norm = jax.lax.sqrt(jax.lax.reduce_sum_p.bind(data_sq, axes=axes) / data.size)
    # Make sure we don't "underflow" too much on the norm.
    norm = jax.lax.max(pow2_round(norm), eps).astype(scale.dtype)
    # Rebalancing based on norm.
    return rebalance(arr, norm)


# Dynamic rescale on fwd arrays.
dynamic_rescale_max = partial(fn_fwd_identity_bwd, dynamic_rescale_max_base)
dynamic_rescale_l1 = partial(fn_fwd_identity_bwd, dynamic_rescale_l1_base)
dynamic_rescale_l2 = partial(fn_fwd_identity_bwd, dynamic_rescale_l2_base)

# Dynamic rescale on gradients.
dynamic_rescale_max_grad = partial(fn_bwd_identity_fwd, dynamic_rescale_max_base)
dynamic_rescale_l1_grad = partial(fn_bwd_identity_fwd, dynamic_rescale_l1_base)
dynamic_rescale_l2_grad = partial(fn_bwd_identity_fwd, dynamic_rescale_l2_base)
