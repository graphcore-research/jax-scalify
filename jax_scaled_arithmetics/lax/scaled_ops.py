# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from jax_scaled_arithmetics import core
from jax_scaled_arithmetics.core import ScaledArray


@core.register_scaled_lax_op
def scaled_mul_p(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)


__all__ = ["scaled_mul_p"]
