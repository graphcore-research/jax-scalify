# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from .typing import Array


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
