# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from jax import lax

from jax_scaled_arithmetics import core
from jax_scaled_arithmetics.core import ScaledArray


def scaled_mul_p(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)


core.register_scaled_op(lax.mul_p, scaled_mul_p)

__all__ = ["scaled_mul_p"]
