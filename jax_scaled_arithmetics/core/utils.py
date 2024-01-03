# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from enum import IntEnum
from typing import Any, Dict

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
