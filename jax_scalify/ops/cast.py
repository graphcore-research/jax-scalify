# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax
import ml_dtypes

from jax_scalify.core import Array, DTypeLike

from .utils import map_on_backward, map_on_forward


def reduce_precision_dtype_base(arr: Array, dtype: DTypeLike) -> Array:
    """`Fake` cast to an ML dtype (e.g. FP8), using JAX LAX `reduce_precision` operator."""
    info = ml_dtypes.finfo(dtype)
    return jax.lax.reduce_precision(arr, exponent_bits=info.nexp, mantissa_bits=info.nmant)


def reduce_precision_on_forward(arr: Array, dtype: DTypeLike) -> Array:
    """`Fake` cast to an ML dtype, on the forward pass (no-op on backward pass)."""
    return partial(map_on_forward, lambda v: reduce_precision_dtype_base(v, dtype))(arr)


def reduce_precision_on_backward(arr: Array, dtype: DTypeLike) -> Array:
    """`Fake` cast to an ML dtype on the backward pass (no-op on forward pass)."""
    return partial(map_on_backward, lambda v: reduce_precision_dtype_base(v, dtype))(arr)


def cast_on_forward(arr: Array, dtype: DTypeLike) -> Array:
    """Cast input array only on the forward pass (no-op on the backward pass).

    Useful for implementation `DenseGeneral` FP8 matmuls.
    """
    return partial(map_on_forward, lambda v: jax.lax.convert_element_type(v, dtype))(arr)


def cast_on_backward(arr: Array, dtype: DTypeLike) -> Array:
    """Cast input array only on the backward pass (no-op on the forward pass).

    Useful for implementation `DenseGeneral` FP8 matmuls.
    """
    return partial(map_on_backward, lambda v: jax.lax.convert_element_type(v, dtype))(arr)
