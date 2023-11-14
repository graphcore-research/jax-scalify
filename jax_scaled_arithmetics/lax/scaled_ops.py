# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from jax import lax

from jax_scaled_arithmetics import core
from jax_scaled_arithmetics.core import ScaledArray
import math


def scaled_mul_p(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)

def scaled_add(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    output_scale = lax.sqrt(A.scale ** 2 + B.scale ** 2)
    output_data = (A.scale/output_scale)*A.data + (B.scale/output_scale)*B.data #check correct type output if mismatch between data and scale precision
    return ScaledArray(output_data, output_scale)

def scaled_abs(A: ScaledArray) -> ScaledArray:
    # Can we assume zero mean gaussian for scale here? Output will therefore not be zero mean - is this a problem? 
    # output_scale = A.scale*lax.sqrt((1 - 2/math.pi))  ---> std of half normal distribution?
    return ScaledArray(lax.abs(A.data), A.scale)

def scaled_div(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data / B.data, A.scale / B.scale)

def scaled_dot(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    output_scale = A.scale * B.scale * lax.sqrt(A.data.shape[-1])
    output_data = lax.dot(A.data, B.data) / lax.sqrt(A.data.shape[-1])
    return ScaledArray(output_data, output_scale)

def scaled_broadcast(A: ScaledArray, sizes) -> ScaledArray:
    return ScaledArray(lax.broadcast(A.data, sizes), A.scale)

def scaled_pow(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    output_data = lax.pow(lax.pow(A.data, B.data), B.scale) #I think doing in this order means we won't overflow
    output_scale = lax.pow(lax.pow(A.scale, B.data), B.scale)
    return ScaledArray(output_data, output_scale)

def scaled_integer_pow(A: ScaledArray, B: int) -> ScaledArray:
    return ScaledArray(A.data ** B, A.scale ** B)

def scaled_sub(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    output_scale = lax.sqrt(A.scale ** 2 + B.scale ** 2)
    output_data = (A.scale/output_scale)*A.data - (B.scale/output_scale)*B.data
    return ScaledArray(output_data, output_scale)

def scaled_square(A: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data ** 2, A.scale ** 2)

def scaled_sqrt(A: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data ** (1/2), A.scale ** (1/2))

def scaled_exp(A: ScaledArray) -> ScaledArray:
    return 

def scaled_log(A: ScaledArray) -> ScaledArray:
    pass

def scaled_max(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    pass

def scaled_min(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    pass




core.register_scaled_op(lax.mul_p, scaled_mul_p)

__all__ = ["scaled_mul_p"]
