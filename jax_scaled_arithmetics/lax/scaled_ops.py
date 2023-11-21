# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Optional, Sequence, Tuple

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


@core.register_scaled_lax_op
def scaled_dot_general(
    lhs: ScaledArray,
    rhs: ScaledArray,
    dimension_numbers: Tuple[Tuple[Sequence[int], Sequence[int]], Tuple[Sequence[int], Sequence[int]]],
    precision: Any = None,
    preferred_element_type: Optional[DTypeLike] = None,
) -> ScaledArray:
    # Checks on `dot_general` arguments. Only supporting a subset right now.
    ((lhs_contracting_dims, rhs_contracting_dims), (lhs_batch_dims, rhs_batch_dims)) = dimension_numbers
    assert len(lhs_batch_dims) == 0
    assert len(rhs_batch_dims) == 0
    assert len(lhs_contracting_dims) == 1
    assert len(rhs_contracting_dims) == 1

    contracting_dim_size = lhs.shape[lhs_contracting_dims[0]]
    # "unit scaling" rule, based on the contracting axis.
    contracting_rescale = np.sqrt(contracting_dim_size).astype(lhs.dtype)
    output_scale = lhs.scale * rhs.scale * contracting_rescale
    output_data = (
        lax.dot_general(
            lhs.data,
            rhs.data,
            dimension_numbers=dimension_numbers,
            precision=precision,
            preferred_element_type=preferred_element_type,
        )
        / contracting_rescale
    )
    return ScaledArray(output_data, output_scale)


@core.register_scaled_lax_op
def scaled_reduce_sum(val: ScaledArray, axes: Tuple[int]) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    shape = val.shape
    axes_size = np.array([shape[idx] for idx in axes])
    # Rescale data component following reduction axes.
    axes_rescale = np.sqrt(np.prod(axes_size))
    data = lax.reduce_sum_p.bind(val.data, axes=axes) / axes_rescale
    outscale = val.scale * axes_rescale
    return ScaledArray(data, outscale)


@core.register_scaled_lax_op
def scaled_reduce_prod(val: ScaledArray, axes: Tuple[int]) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    shape = val.shape
    data = lax.reduce_prod_p.bind(val.data, axes=axes)
    axes_size = np.prod(np.array([shape[idx] for idx in axes]))
    scale = lax.integer_pow(val.scale, axes_size)
    return ScaledArray(data, scale)


@core.register_scaled_lax_op
def scaled_reduce_max(val: ScaledArray, axes: Tuple[int]) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    data = lax.reduce_max_p.bind(val.data, axes=axes)
    # unchanged scaling.
    return ScaledArray(data, val.scale)


@core.register_scaled_lax_op
def scaled_reduce_min(val: ScaledArray, axes: Tuple[int]) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    data = lax.reduce_min_p.bind(val.data, axes=axes)
    # unchanged scaling.
    return ScaledArray(data, val.scale)
