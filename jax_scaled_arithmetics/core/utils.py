# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from enum import IntEnum
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from numpy.typing import NDArray

from .typing import Array, get_numpy_api

# Exponent bits masking.
_exponent_bits_mask: Dict[Any, NDArray[Any]] = {
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


class Pow2RoundMode(IntEnum):
    """Power-of-two supported rounded mode."""

    NONE = 0
    DOWN = 1
    UP = 2
    STOCHASTIC = 3


def get_mantissa(val: Array) -> Array:
    """Extract the mantissa of an array, masking the exponent.

    Similar to `numpy.frexp`, but with implicit bit to be consistent with
    `pow2_round_down`.
    """
    np_api = get_numpy_api(val)
    # TODO: implement using bitmasking?
    mantissa_val, _ = np_api.frexp(val)
    # Re-add the implicit bit to be consistent with `pow2_round_down`
    mantissa_val = mantissa_val * np.array(2, dtype=val.dtype)
    return mantissa_val


def pow2_round_down(val: Array) -> Array:
    """Round down to the closest power of 2."""
    np_api = get_numpy_api(val)
    exponent_mask = _exponent_bits_mask[val.dtype]
    intdtype = exponent_mask.dtype
    pow2_val = np_api.bitwise_and(val.view(intdtype), exponent_mask).view(val.dtype).reshape(val.shape)
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


def safe_div(lhs: Array, rhs: Array) -> Array:
    """Safe (scalar) div: if rhs is zero, returns zero."""
    assert lhs.shape == ()
    assert rhs.shape == ()
    # assert lhs.dtype == rhs.dtype
    # Numpy inputs => direct computation.
    is_npy_inputs = isinstance(lhs, (np.number, np.ndarray)) and isinstance(rhs, (np.number, np.ndarray))
    if is_npy_inputs:
        return np.divide(lhs, rhs, out=np.array(0, dtype=rhs.dtype), where=rhs != 0)
    # JAX general implementation.
    return jax.lax.select(rhs == 0, rhs, jnp.divide(lhs, rhs))


def safe_reciprocal(val: Array) -> Array:
    """Safe (scalar) reciprocal: if val is zero, returns zero."""
    assert val.shape == ()
    # Numpy inputs => direct computation.
    if isinstance(val, (np.number, np.ndarray)):
        return np.reciprocal(val, out=np.array(0, dtype=val.dtype), where=val != 0)
    # JAX general implementation.
    return jax.lax.select(val == 0, val, jax.lax.reciprocal(val))


def python_scalar_as_numpy(val: Any) -> Any:
    """Convert Python scalar to Numpy scalar, if possible.

    Using by default JAX 32 bits precision, instead of 64 bits.

    Returning unchanged value if not any (bool, int, float).
    """
    if isinstance(val, bool):
        return np.bool_(val)
    elif isinstance(val, int):
        return np.int32(val)
    elif isinstance(val, float):
        return np.float32(val)
    return val
