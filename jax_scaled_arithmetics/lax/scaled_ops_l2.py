# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Optional, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import lax
from jax._src.ad_util import add_any_p

from jax_scaled_arithmetics import core
from jax_scaled_arithmetics.core import (
    DTypeLike,
    ScaledArray,
    as_scaled_array,
    get_autoscale_config,
    pow2_round,
    register_scaled_op,
    safe_div,
)

from .scaled_ops_common import check_scalar_scales, promote_scale_types


def scaled_add_sub(A: ScaledArray, B: ScaledArray, binary_op: Any) -> ScaledArray:
    """Scaled add/sub generic implementation."""
    # TODO: understand when promotion is really required?
    A, B = as_scaled_array((A, B))  # type:ignore
    check_scalar_scales(A, B)
    A, B = promote_scale_types(A, B)
    assert np.issubdtype(A.scale.dtype, np.floating)
    # Pow2 rounding for unit scaling "rule".
    pow2_rounding_mode = get_autoscale_config().rounding_mode
    # TODO: what happens to `sqrt` for non-floating scale?
    # More stable than direct L2 norm, to avoid scale overflow.
    ABscale_max = lax.max(A.scale, B.scale)
    ABscale_min = lax.min(A.scale, B.scale)
    ABscale_ratio = safe_div(ABscale_min, ABscale_max)
    output_scale = ABscale_max * lax.sqrt(1 + ABscale_ratio * ABscale_ratio)
    # Transform back to power-of-2
    output_scale = pow2_round(output_scale, pow2_rounding_mode)
    # Output dtype => promotion of A and B dtypes.
    outdtype = jnp.promote_types(A.dtype, B.dtype)
    Arescale = safe_div(A.scale, output_scale).astype(outdtype)
    Brescale = safe_div(B.scale, output_scale).astype(outdtype)
    # check correct type output if mismatch between data and scale precision
    output_data = binary_op(Arescale * A.data, Brescale * B.data)
    return ScaledArray(output_data, output_scale)


@core.register_scaled_lax_op
def scaled_add(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return scaled_add_sub(A, B, lax.add)


# TODO: understand difference between `add` and `add_anys`
register_scaled_op(add_any_p, scaled_add)


@core.register_scaled_lax_op
def scaled_sub(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    return scaled_add_sub(A, B, lax.sub)


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

    # Pow2 rounding for unit scaling "rule".
    pow2_rounding_mode = get_autoscale_config().rounding_mode
    contracting_dim_size = lhs.shape[lhs_contracting_dims[0]]
    # "unit scaling" rule, based on the contracting axis.
    outscale_dtype = jnp.promote_types(lhs.scale.dtype, rhs.scale.dtype)
    contracting_rescale = pow2_round(np.sqrt(contracting_dim_size), pow2_rounding_mode)
    # Keeping power of 2 scale.
    output_scale = lhs.scale * rhs.scale * contracting_rescale.astype(outscale_dtype)
    # NOTE: need to be a bit careful about scale promotion?
    output_data = lax.dot_general(
        lhs.data,
        rhs.data,
        dimension_numbers=dimension_numbers,
        precision=precision,
        preferred_element_type=preferred_element_type,
    )
    output_data = output_data / contracting_rescale.astype(output_data.dtype)
    return ScaledArray(output_data, output_scale)


@core.register_scaled_lax_op
def scaled_conv_general_dilated(lhs: ScaledArray, rhs: ScaledArray, **params) -> ScaledArray:
    assert isinstance(lhs, ScaledArray)
    assert isinstance(rhs, ScaledArray)
    data = lax.conv_general_dilated_p.bind(lhs.data, rhs.data, **params)
    # FIXME: should we change scaling if e.g. window > 3?
    return ScaledArray(data, lhs.scale * rhs.scale)


@core.register_scaled_lax_op
def scaled_reduce_sum(val: ScaledArray, axes: Tuple[int]) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    shape = val.shape
    axes_size = np.array([shape[idx] for idx in axes])
    # Pow2 rounding for unit scaling "rule".
    pow2_rounding_mode = get_autoscale_config().rounding_mode
    # Rescale data component following reduction axes & round to power of 2 value.
    axes_rescale = np.sqrt(np.prod(axes_size))
    axes_rescale = pow2_round(axes_rescale, pow2_rounding_mode)
    data = lax.reduce_sum_p.bind(val.data, axes=axes) / axes_rescale.astype(val.data.dtype)
    outscale = val.scale * axes_rescale.astype(val.scale.dtype)
    return ScaledArray(data, outscale)


@core.register_scaled_lax_op
def scaled_reduce_prod(val: ScaledArray, axes: Tuple[int]) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    shape = val.shape
    data = lax.reduce_prod_p.bind(val.data, axes=axes)
    axes_size = np.prod(np.array([shape[idx] for idx in axes]))
    # Stable for power of 2.
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


@core.register_scaled_lax_op
def scaled_reduce_window_sum(
    val: ScaledArray,
    window_dimensions: Any,
    window_strides: Any,
    padding: Any,
    base_dilation: Any,
    window_dilation: Any,
) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    data = lax.reduce_window_sum_p.bind(
        val.data,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
        base_dilation=base_dilation,
        window_dilation=window_dilation,
    )
    # FIXME: should we change scaling if e.g. window > 3?
    return ScaledArray(data, val.scale)


@core.register_scaled_lax_op
def scaled_reduce_window_min(
    val: ScaledArray,
    window_dimensions: Any,
    window_strides: Any,
    padding: Any,
    base_dilation: Any,
    window_dilation: Any,
) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    data = lax.reduce_window_min_p.bind(
        val.data,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
        base_dilation=base_dilation,
        window_dilation=window_dilation,
    )
    # unchanged scaling.
    return ScaledArray(data, val.scale)


@core.register_scaled_lax_op
def scaled_reduce_window_max(
    val: ScaledArray,
    window_dimensions: Any,
    window_strides: Any,
    padding: Any,
    base_dilation: Any,
    window_dilation: Any,
) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    data = lax.reduce_window_max_p.bind(
        val.data,
        window_dimensions=window_dimensions,
        window_strides=window_strides,
        padding=padding,
        base_dilation=base_dilation,
        window_dilation=window_dilation,
    )
    # unchanged scaling.
    return ScaledArray(data, val.scale)
