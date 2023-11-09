# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, Sequence

import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_scaled_arithmetics import core
from jax_scaled_arithmetics.core import DTypeLike, ScaledArray, Shape


@core.register_scaled_lax_op
def scaled_broadcast_in_dim(A: ScaledArray, shape: Shape, broadcast_dimensions: Sequence[int]) -> ScaledArray:
    return ScaledArray(lax.broadcast_in_dim(A.data, shape=shape, broadcast_dimensions=broadcast_dimensions), A.scale)


@core.register_scaled_lax_op
def scaled_convert_element_type(A: ScaledArray, new_dtype: DTypeLike, weak_type: bool = False) -> ScaledArray:
    # NOTE: by default, no rescaling done before casting.
    # Choice of adding an optional rescaling op before is up to the user (and which strategy to use).
    # NOTE bis: scale not casted as well by default!
    return ScaledArray(lax.convert_element_type(A.data, new_dtype=new_dtype), A.scale)


@core.register_scaled_lax_op
def scaled_concatenate(operands: Sequence[ScaledArray], dimension: int) -> ScaledArray:
    # TODO: inputs checking (dtype and cie).
    scales = jnp.array([v.scale for v in operands])
    # Max rescaling of the collection of operands.
    # TODO: explore alternative strategies?
    scale_max = jnp.max(scales)
    datas = [v.data * (v.scale / scale_max) for v in operands]
    data_concat = lax.concatenate(datas, dimension=dimension)
    return ScaledArray(data_concat, scale_max)


@core.register_scaled_lax_op
def scaled_slice(
    A: ScaledArray, start_indices: Sequence[int], limit_indices: Sequence[int], strides: Optional[Sequence[int]] = None
) -> ScaledArray:
    return ScaledArray(
        lax.slice(A.data, start_indices=start_indices, limit_indices=limit_indices, strides=strides), A.scale
    )


@core.register_scaled_lax_op
def scaled_transpose(A: ScaledArray, permutation: Sequence[int]) -> ScaledArray:
    return ScaledArray(lax.transpose(A.data, permutation=permutation), A.scale)


@core.register_scaled_lax_op
def scaled_mul(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return ScaledArray(A.data * B.data, A.scale * B.scale)


@core.register_scaled_lax_op
def scaled_add(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    # Only supporting floating scale right now.
    assert A.scale.dtype == B.scale.dtype
    assert np.issubdtype(A.scale, np.floating)
    # TODO: what happens to `sqrt` for non-floating scale?
    output_scale = lax.sqrt(A.scale**2 + B.scale**2)
    # check correct type output if mismatch between data and scale precision
    output_data = (A.scale / output_scale) * A.data + (B.scale / output_scale) * B.data
    return ScaledArray(output_data, output_scale)


@core.register_scaled_lax_op
def scaled_sub(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    # Only supporting floating scale right now.
    assert A.scale.dtype == B.scale.dtype
    assert np.issubdtype(A.scale, np.floating)
    # TODO: what happens to `sqrt` for non-floating scale?
    output_scale = lax.sqrt(A.scale**2 + B.scale**2)
    # check correct type output if mismatch between data and scale precision
    output_data = (A.scale / output_scale) * A.data - (B.scale / output_scale) * B.data
    return ScaledArray(output_data, output_scale)


def scaled_dot(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    output_scale = A.scale * B.scale * lax.sqrt(A.data.shape[-1])
    output_data = lax.dot(A.data, B.data) / lax.sqrt(A.data.shape[-1])
    return ScaledArray(output_data, output_scale)
