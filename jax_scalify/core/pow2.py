# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import logging
from enum import IntEnum
from functools import partial
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import jax.numpy as jnp
import ml_dtypes
import numpy as np
from jax import core
from jax.interpreters import mlir
from jax.interpreters.mlir import LoweringRuleContext, ir
from numpy.typing import DTypeLike, NDArray

from .typing import Array, get_numpy_api

# Exponent bits masking.
_exponent_bits_mask: Dict[Any, NDArray[Any]] = {
    np.dtype(jnp.bfloat16): np.packbits(
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    ).view(np.int16),
    # Copy for ml_dtypes.bfloat16, distinct in older JAX versions.
    np.dtype(ml_dtypes.bfloat16): np.packbits(
        np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1], dtype=np.uint8)
    ).view(np.int16),
    np.dtype(np.float16): np.packbits(np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0], dtype=np.uint8)).view(
        np.int16
    ),
    np.dtype(np.float32): np.packbits(
        np.array(
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1],
            dtype=np.uint8,
        )
    ).view(np.int32),
    np.dtype(np.float64): np.array(np.inf, np.float64).view(np.int64),
}
"""Exponents bit masking: explicit bitmask to keep only exponent bits in floating point values.

NOTE: normally should also correspond to `np.inf` value for FP16 and FP32.
"""


def dtype_exponent_mask(dtype: DTypeLike, sign_bit: bool = False) -> NDArray[Any]:
    """Get the exponent mask for a given Numpy/JAX dtype.

    Args:
        dtype: Numpy/JAX dtype.
        sign_bit: Include sign bit in the mask.
    Returns:
        Array mask as integer dtype.
    """
    mask = _exponent_bits_mask[dtype]
    if sign_bit:
        # Negative value to add sign.
        intdtype = mask.dtype
        mask = (-mask.view(dtype)).view(intdtype)
        return mask
    return mask


def pow2_decompose_round_down_impl(vin: Array, scale_dtype: DTypeLike) -> Array:
    """Pow-2 decompose with rounding down.

    Returns:
        (scale, vout) such that vin = scale * vout
    """
    np_api = get_numpy_api(vin)
    # Perform all computations in FP32, to support FP16 submormals.
    # NOTE: `jnp.frexp` is buggy for subnormals.
    dtype = np.dtype(np.float32)
    minval = np.finfo(dtype).smallest_normal
    exponent_mask = dtype_exponent_mask(dtype)
    intdtype = exponent_mask.dtype
    val = vin.astype(dtype)
    # Masking mantissa bits, keeping only the exponents ones.
    scale_pow2 = np_api.bitwise_and(val.view(intdtype), exponent_mask).view(val.dtype).reshape(val.shape)
    # Get the mantissa in float32. Make sure we don't divide by zero, and handle nan/inf.
    normal_scale_val = np_api.logical_and(np_api.isfinite(scale_pow2), scale_pow2 != 0)
    scale_renorm = np_api.where(normal_scale_val, scale_pow2, minval)
    mantissa = val / scale_renorm
    return scale_pow2.astype(scale_dtype), mantissa.astype(vin.dtype)


class Pow2RoundMode(IntEnum):
    """Power-of-two supported rounded mode."""

    NONE = 0
    DOWN = 1
    UP = 2
    STOCHASTIC = 3


pow2_decompose_p = core.Primitive("pow2_decompose")
"""`pow2_decompose` pow2 decompose JAX primitive.
"""


def pow2_decompose(
    vin: Array, scale_dtype: Optional[DTypeLike] = None, mode: Pow2RoundMode = Pow2RoundMode.DOWN
) -> Tuple[Array, Array]:
    """Power-2 decompose, i.e. vin = s * vout where s is a power-of 2 scaling.

    Args:
        vin: Input array.
        scale_dtype: Scale dtype to use.
        mode: Pow2 rounding.
    Returns:
        (scale, vout) such that vin = scale * vout
    """
    scale_dtype = np.dtype(scale_dtype or vin.dtype)
    # A couple of checks on dtypes.
    assert np.issubdtype(vin.dtype, np.floating)
    assert np.issubdtype(scale_dtype, np.floating)
    if scale_dtype == np.float16:
        logging.warning("`pow2_decompose` does not support FP16 sub-normals when using FP16 scale dtype.")
    out = pow2_decompose_p.bind(vin, scale_dtype=scale_dtype, mode=mode)
    return out


def pow2_decompose_eager_impl(
    vin: Array, scale_dtype: Optional[DTypeLike] = None, mode: Pow2RoundMode = Pow2RoundMode.DOWN
) -> Tuple[Array, Array]:
    """Eager mode implementation, on JAX/Numpy arrays."""
    if mode == Pow2RoundMode.DOWN:
        return pow2_decompose_round_down_impl(vin, scale_dtype)
    raise NotImplementedError(f"Unsupported power-of-2 rounding mode '{mode}'.")


def pow2_decompose_abstract_eval(
    vin: core.ShapedArray, scale_dtype: Optional[DTypeLike] = None, mode: Pow2RoundMode = Pow2RoundMode.DOWN
) -> Tuple[core.ShapedArray, core.ShapedArray]:
    scale_dtype = scale_dtype or vin.dtype
    sout = core.ShapedArray(vin.shape, dtype=scale_dtype)
    return (sout, vin)


def pow2_decompose_mlir_lowering(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]], **params: Dict[str, Any]
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    scale_dtype = params["scale_dtype"]
    mode = params["mode"]
    pow2_decompose_fn = partial(pow2_decompose_eager_impl, scale_dtype=scale_dtype, mode=mode)
    outputs = mlir.lower_fun(pow2_decompose_fn, multiple_results=True)(ctx, *args)
    return outputs


# Register as standard JAX primitive
pow2_decompose_p.multiple_results = True
pow2_decompose_p.def_abstract_eval(pow2_decompose_abstract_eval)
pow2_decompose_p.def_impl(pow2_decompose_eager_impl)
# Default lowering on GPU, TPU, ...
mlir.register_lowering(pow2_decompose_p, pow2_decompose_mlir_lowering)


def pow2_round_down(val: Array) -> Array:
    """Round down to the closest power of 2."""
    # Keep only the scale component of `pow2_decompose`
    pow2_val, _ = pow2_decompose(val, scale_dtype=val.dtype, mode=Pow2RoundMode.DOWN)
    return pow2_val


def pow2_round_up(val: Array) -> Array:
    """Round up to the closest power of 2.
    NOTE: may overflow to inf.
    """
    # FIXME: rounding when already a power of 2.
    # Should do additional masking to check that.
    pow2_val = pow2_round_down(val) * np.array(2, dtype=val.dtype)
    return pow2_val


def pow2_round(val: Array, mode: Pow2RoundMode = Pow2RoundMode.DOWN) -> Array:
    """Power-of-two rounding."""
    if mode == Pow2RoundMode.NONE:
        return val
    elif mode == Pow2RoundMode.DOWN:
        return pow2_round_down(val)
    elif mode == Pow2RoundMode.UP:
        return pow2_round_up(val)
    raise NotImplementedError(f"Unsupported power-of-2 rounding mode '{mode}'.")


def get_mantissa(val: Array) -> Array:
    """Extract the mantissa of an array, masking the exponent.

    Similar to `numpy.frexp`, but with implicit bit to be consistent with
    `pow2_round_down`.
    """
    _, mantissa = pow2_decompose(val, scale_dtype=val.dtype, mode=Pow2RoundMode.DOWN)
    return mantissa
