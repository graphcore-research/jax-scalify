# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import jax
import ml_dtypes

from jax_scalify.core import Array, DTypeLike

from .rescaling import fn_bwd_identity_fwd, fn_fwd_identity_bwd


def reduce_precision_dtype_base(arr: Array, dtype: DTypeLike) -> Array:
    """`Fake` cast to an ML dtype (e.g. FP8), using JAX LAX `reduce_precision` operator."""
    info = ml_dtypes.finfo(dtype)
    return jax.lax.reduce_precision(arr, exponent_bits=info.nexp, mantissa_bits=info.nmant)


def reduce_precision_dtype(arr: Array, dtype: DTypeLike) -> Array:
    """`Fake` cast to an ML dtype, on the forward pass (no-op on backward pass)."""
    return partial(fn_fwd_identity_bwd, lambda v: reduce_precision_dtype_base(v, dtype))(arr)


def reduce_precision_dtype_grad(arr: Array, dtype: DTypeLike) -> Array:
    """`Fake` cast to an ML dtype on the backward pass (no-op on forward pass)."""
    return partial(fn_bwd_identity_fwd, lambda v: reduce_precision_dtype_base(v, dtype))(arr)
