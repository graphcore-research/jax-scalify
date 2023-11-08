from core import ScaledArray
import core
from jax import lax
from functools import partial


# Tried as decorator too


# @partial(core.register_scaled_op, lax_func=lax.mul)
def scaled_mul_p(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)


core.register_scaled_op(lax.mul_p, scaled_mul_p)

__all__ = ["scaled_mul_p"]
