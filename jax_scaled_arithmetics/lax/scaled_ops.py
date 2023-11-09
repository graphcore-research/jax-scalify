# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from jax import lax

from jax_scaled_arithmetics import core
from jax_scaled_arithmetics.core import ScaledArray
import math


def scaled_mul_p(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)

def scaled_add(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    output_scale = lax.sqrt(A.scale ** 2 + B.scale ** 2)
    output_data = (A.scale/output_scale)*A.data + (B.scale/output_scale)*B.data
    return ScaledArray(output_data, output_scale)

def scaled_abs(A: ScaledArray) -> ScaledArray:
    # Can we assume zero mean gaussian for scale here? Output will therefore not be zero mean - is this a problem? 
    output_scale = A.scale*lax.sqrt((1 - 2/math.pi)) #std of half normal distribution
    return ScaledArray(lax.abs(A.data), output_scale)

def scaled_sqrt(A: ScaledArray) -> ScaledArray:
    pass

def scaled_cbrt(A: ScaledArray) -> ScaledArray:
    pass

def scaled_div(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)

def scaled_dot(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    pass

def scaled_erf(A: ScaledArray) -> ScaledArray:
    pass

def scaled_exp(A: ScaledArray) -> ScaledArray:
    pass

def scaled_log(A: ScaledArray) -> ScaledArray:
    pass

def scaled_max(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    pass

def scaled_min(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    pass

def scaled_pow(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    pass

def scaled_sub(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    output_scale = lax.sqrt(A.scale ** 2 + B.scale ** 2)
    output_data = (A.scale/output_scale)*A.data - (B.scale/output_scale)*B.data
    return ScaledArray(output_data, output_scale)



core.register_scaled_op(lax.mul_p, scaled_mul_p)

__all__ = ["scaled_mul_p"]
