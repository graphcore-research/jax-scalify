# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import jax.numpy as jnp
import ml_dtypes
import numpy as np

from jax_scalify.core import Array, DTypeLike, get_numpy_api
from jax_scalify.core.pow2 import dtype_exponent_mask


def pow2_truncate(arr: Array) -> Array:
    """Convert an Array to a power of 2, using mantissa truncation.

    NOTE: all sub-normals values are flushed to zero.
    """
    np_api = get_numpy_api(arr)
    # Masking mantissa & sign-bit, keeping only exponent values.
    exponent_mask = dtype_exponent_mask(arr.dtype, sign_bit=True)
    intdtype = exponent_mask.dtype
    # Masking mantissa bits, keeping only the exponents ones.
    arr_pow2 = np_api.bitwise_and(arr.view(intdtype), exponent_mask).view(arr.dtype).reshape(arr.shape)
    return arr_pow2


def as_e8m0(arr: Array) -> Array:
    """Convert an Array to e8m0 format (i.e. power of two values).

    This function is only implementing a truncation + saturation variant, in line with
    the MX OCP format.

    Args:
        arr: Input array (FP16, FP32 or BF16).
    Returns:
        E8M0 array (as uint8).
    """
    np_api = get_numpy_api(arr)
    # assert len(arr.shape) < 2
    assert arr.dtype in {np.dtype(jnp.bfloat16), np.dtype(ml_dtypes.bfloat16), np.dtype(jnp.float32)}
    # Saturation => negative values saturating to min value (i.e. zero bits) in E8M0.
    arr = np_api.maximum(arr, np.array(0, arr.dtype))
    arr = pow2_truncate(arr)

    # Bit masking to extract the exponent as uint8 array.
    arr_u8 = arr.view(np.uint8).reshape((*arr.shape, -1))
    arr_e8m0 = np_api.bitwise_or(np_api.left_shift(arr_u8[..., -1], 1), np_api.right_shift(arr_u8[..., -2], 7))
    return arr_e8m0


def from_e8m0(arr: Array, dtype: DTypeLike) -> Array:
    """Convert an Array of e8m0 values (i.e. power of two values) to a given dtype.

    Args:
        arr: E8M0 array (assuming uint8 storage dtype).
        dtype: Output dtype. FP32 or BF16 supported.
    Returns:
        Converted output.
    """
    np_api = get_numpy_api(arr)
    assert arr.dtype == np.uint8
    assert np.dtype(dtype) in {np.dtype(jnp.bfloat16), np.dtype(ml_dtypes.bfloat16), np.dtype(jnp.float32)}
    # Avoid issues with 7 mantissa bits in BF16.
    # TODO: more efficient implementation!
    arr = np_api.exp2(arr.astype(np.float32) - 127)
    return arr.astype(dtype)
