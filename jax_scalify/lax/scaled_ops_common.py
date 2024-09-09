# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Optional, Sequence

import jax
import jax.core
import jax.numpy as jnp
import numpy as np
from jax import lax

from jax_scalify import core
from jax_scalify.core import (
    Array,
    DTypeLike,
    ScaledArray,
    Shape,
    Sharding,
    as_scaled_array,
    get_scale_dtype,
    is_static_anyscale,
    is_static_zero,
    safe_div,
)

from .base_scaling_primitives import scaled_set_scaling


def _get_data(val: Any) -> Array:
    if isinstance(val, ScaledArray):
        return val.data
    return val


def check_scalar_scales(*args: ScaledArray) -> Array:
    """Check all ScaledArrays have scalar scaling."""
    for val in args:
        assert np.ndim(val.scale) == 0


def promote_types(*args: DTypeLike) -> DTypeLike:
    """Find a common promotion dtype."""
    outdtype = args[0]
    for val in args[1:]:
        outdtype = jnp.promote_types(outdtype, val)
    return outdtype


def promote_scale_types(*args: ScaledArray) -> Sequence[ScaledArray]:
    """Promote scale datatypes to a common one.

    Note: we are using JAX Numpy promotion, to avoid 64bits types by default.
    """
    if len(args) == 1:
        return args
    # Find a common scale datatype.
    scale_dtype = args[0].scale.dtype
    for val in args[1:]:
        scale_dtype = jnp.promote_types(scale_dtype, val.scale.dtype)

    outputs = [ScaledArray(v.data, v.scale.astype(scale_dtype)) for v in args]
    return outputs


@core.register_scaled_lax_op
def scaled_stop_gradient(val: ScaledArray) -> ScaledArray:
    # Stop gradients on both data and scale tensors.
    return ScaledArray(lax.stop_gradient(val.data), lax.stop_gradient(val.scale))


@core.register_scaled_lax_op
def scaled_reshape(A: ScaledArray, new_sizes: Sequence[int], dimensions: Optional[Sequence[int]]) -> ScaledArray:
    return ScaledArray(lax.reshape(A.data, new_sizes=new_sizes, dimensions=dimensions), A.scale)


@core.register_scaled_lax_op
def scaled_broadcast_in_dim(A: ScaledArray, shape: Shape, broadcast_dimensions: Sequence[int]) -> ScaledArray:
    return ScaledArray(lax.broadcast_in_dim(A.data, shape=shape, broadcast_dimensions=broadcast_dimensions), A.scale)


@core.register_scaled_lax_op
def scaled_convert_element_type(
    A: ScaledArray, new_dtype: DTypeLike, weak_type: bool = False, sharding: Sharding | None = None
) -> ScaledArray:
    # NOTE: by default, no rescaling done before casting.
    # Choice of adding an optional rescaling op before is up to the user (and which strategy to use).
    # NOTE bis: scale not casted as well by default!
    return ScaledArray(lax.convert_element_type(A.data, new_dtype=new_dtype), A.scale)


@core.register_scaled_lax_op
def scaled_reduce_precision(A: ScaledArray, exponent_bits: int, mantissa_bits: int) -> ScaledArray:
    # Applying precision reduction only data term.
    return ScaledArray(lax.reduce_precision(A.data, exponent_bits=exponent_bits, mantissa_bits=mantissa_bits), A.scale)


@core.register_scaled_lax_op
def scaled_concatenate(operands: Sequence[ScaledArray], dimension: int) -> ScaledArray:
    # TODO: inputs checking (dtype and cie).
    scales = jnp.array([v.scale for v in operands])
    # Max rescaling of the collection of operands. Preserving pow2 scaling.
    # TODO: explore alternative strategies?
    outdtype = operands[0].dtype
    scale_max = jnp.max(scales)
    datas = [v.data * safe_div(v.scale, scale_max).astype(outdtype) for v in operands]
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
def scaled_rev(val: ScaledArray, dimensions: Sequence[int]) -> ScaledArray:
    return ScaledArray(lax.rev(val.data, dimensions=dimensions), val.scale)


@core.register_scaled_lax_op
def scaled_pad(val: ScaledArray, padding_value: Any, padding_config: Any) -> ScaledArray:
    # Only supporting constant zero padding for now.
    assert np.all(is_static_zero(padding_value))
    return ScaledArray(lax.pad(val.data, np.array(0, val.dtype), padding_config), val.scale)


@core.register_scaled_lax_op
def scaled_argmax(val: ScaledArray, axes: Sequence[int], index_dtype: DTypeLike) -> Array:
    # Note: returning a normal `int` Array.
    return lax.argmax_p.bind(val.data, axes=axes, index_dtype=index_dtype)


@core.register_scaled_lax_op
def scaled_argmin(val: ScaledArray, axes: Sequence[int], index_dtype: DTypeLike) -> Array:
    # Note: returning a normal `int` Array.
    return lax.argmin_p.bind(val.data, axes=axes, index_dtype=index_dtype)


@core.register_scaled_lax_op
def scaled_neg(val: ScaledArray) -> ScaledArray:
    return ScaledArray(-val.data, val.scale)


@core.register_scaled_lax_op
def scaled_abs(val: ScaledArray) -> ScaledArray:
    return ScaledArray(jax.lax.abs(val.data), val.scale)


@core.register_scaled_lax_op
def scaled_mul(lhs: ScaledArray, rhs: ScaledArray) -> ScaledArray:
    # TODO: understand when promotion is really required?
    lhs, rhs = as_scaled_array((lhs, rhs))  # type:ignore
    return ScaledArray(lhs.data * rhs.data, lhs.scale * rhs.scale)


@core.register_scaled_lax_op
def scaled_div(lhs: ScaledArray, rhs: ScaledArray) -> ScaledArray:
    # TODO: understand when promotion is really required?
    # lhs, rhs = as_scaled_array((lhs, rhs))  # type:ignore
    # TODO: investigate different rule?
    return ScaledArray(lhs.data / rhs.data, lhs.scale / rhs.scale)


@core.register_scaled_lax_op
def scaled_sign(val: ScaledArray) -> Array:
    assert isinstance(val, ScaledArray)
    # Just need to run `lax.sign` on main data.
    return ScaledArray(lax.sign(val.data), np.array(1, dtype=val.scale.dtype))


@core.register_scaled_lax_op
def scaled_is_finite(val: ScaledArray) -> Array:
    assert isinstance(val, ScaledArray)
    if np.issubdtype(val.scale.dtype, np.integer):
        # Integer scale case => only check the data component.
        return lax.is_finite(val.data)
    # Both data & scale need to be finite!
    return lax.and_p.bind(lax.is_finite(val.data), lax.is_finite(val.scale))


def scaled_boolean_binary_op(lhs: ScaledArray, rhs: ScaledArray, prim: jax.core.Primitive) -> Array:
    """Generic implementation of any boolean binary operation."""
    assert isinstance(lhs, ScaledArray)
    assert isinstance(rhs, ScaledArray)
    # FIXME: fix this absolute horror!
    # TODO: use max scale + special case for scalars.
    return prim.bind(lhs.to_array(dtype=np.float32), rhs.to_array(dtype=np.float32))


@core.register_scaled_lax_op
def scaled_eq(lhs: ScaledArray, rhs: ScaledArray) -> Array:
    return scaled_boolean_binary_op(lhs, rhs, lax.eq_p)


@core.register_scaled_lax_op
def scaled_ne(lhs: ScaledArray, rhs: ScaledArray) -> Array:
    return scaled_boolean_binary_op(lhs, rhs, lax.ne_p)


@core.register_scaled_lax_op
def scaled_gt(lhs: ScaledArray, rhs: ScaledArray) -> Array:
    return scaled_boolean_binary_op(lhs, rhs, lax.gt_p)


@core.register_scaled_lax_op
def scaled_ge(lhs: ScaledArray, rhs: ScaledArray) -> Array:
    return scaled_boolean_binary_op(lhs, rhs, lax.ge_p)


@core.register_scaled_lax_op
def scaled_lt(lhs: ScaledArray, rhs: ScaledArray) -> Array:
    return scaled_boolean_binary_op(lhs, rhs, lax.lt_p)


@core.register_scaled_lax_op
def scaled_le(lhs: ScaledArray, rhs: ScaledArray) -> Array:
    return scaled_boolean_binary_op(lhs, rhs, lax.le_p)


def scaled_minmax(prim: jax.core.Primitive, lhs: ScaledArray, rhs: ScaledArray) -> ScaledArray:
    """General min/max scaled translation: propagating the largest input scale."""
    check_scalar_scales(lhs, rhs)
    # Specific rule if lhs/rhs is zero or inf => propagate the other term scale.
    if np.all(is_static_anyscale(lhs)):
        return ScaledArray(prim.bind(lhs.data, rhs.data), rhs.scale)
    if np.all(is_static_anyscale(rhs)):
        return ScaledArray(prim.bind(lhs.data, rhs.data), lhs.scale)

    # Power-of-2 stable!
    output_scale = lax.max(lhs.scale, rhs.scale)
    # TODO: isolate this "binary" rescale logic into separate function.
    outdtype = jnp.promote_types(lhs.dtype, rhs.dtype)
    lhs_rescale = safe_div(lhs.scale, output_scale).astype(outdtype)
    rhs_rescale = safe_div(rhs.scale, output_scale).astype(outdtype)
    output_data = prim.bind(lhs_rescale * lhs.data, rhs_rescale * rhs.data)
    return ScaledArray(output_data, output_scale)


@core.register_scaled_lax_op
def scaled_min(lhs: ScaledArray, rhs: ScaledArray) -> ScaledArray:
    return scaled_minmax(lax.min_p, lhs, rhs)


@core.register_scaled_lax_op
def scaled_max(lhs: ScaledArray, rhs: ScaledArray) -> ScaledArray:
    return scaled_minmax(lax.max_p, lhs, rhs)


##################################################################
# Default scaled ops implementation #
##################################################################
def scaled_op_default_translation(
    prim: jax.core.Primitive, args: Sequence[ScaledArray], outscale: Optional[Array] = None
) -> ScaledArray:
    """Scaled op default translation of a JAX primitive: unscaling inputs + calling normal primitive.

    Args:
        prim: JAX primitive
        args: Input arguments.
        outscale: Output scale to use.
    """
    inputs = [core.asarray(v) for v in args]
    output = prim.bind(*inputs)
    # Rescale output, if necessary.
    if outscale is None:
        output_scale_dtype = promote_types(*[v.scale.dtype for v in args])
        return ScaledArray(output, np.array(1.0, dtype=output_scale_dtype))
    output_scaled = scaled_set_scaling(output, outscale)
    return output_scaled


@core.register_scaled_lax_op
def scaled_exp(val: ScaledArray) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    # Estimate in FP32, to avoid NaN when "descaling" the array.
    # Otherwise: issues for representing properly 0 and +-Inf.
    arr = val.to_array(dtype=np.float32).astype(val.dtype)
    scale = np.array(1, dtype=val.scale.dtype)
    return ScaledArray(lax.exp(arr), scale)


@core.register_scaled_lax_op
def scaled_log(val: ScaledArray) -> ScaledArray:
    assert isinstance(val, ScaledArray)
    # Log of data & scale components.
    log_data = lax.log(val.data)
    log_scale = lax.log(val.scale).astype(val.dtype)
    data = log_data + log_scale
    scale = np.array(1, dtype=val.scale.dtype)
    return ScaledArray(data, scale)


@core.register_scaled_lax_op
def scaled_select_n(which: Array, *cases: ScaledArray) -> ScaledArray:
    outscale_dtype = promote_types(*[get_scale_dtype(v) for v in cases])
    # Get the max scale for renormalizing.
    # TODO: use `get_data_scale` primitive.
    scale_cases = [v.scale if isinstance(v, ScaledArray) else np.array(1, outscale_dtype) for v in cases]
    scales_arr = jnp.array(scale_cases)
    outscale = jnp.max(scales_arr)
    # `data` components, renormalized.
    data_cases = [_get_data(v) * (s / outscale).astype(v.dtype) for s, v in zip(scale_cases, cases)]
    # Apply normal select!
    outdata = lax.select_n(which, *data_cases)
    return ScaledArray(outdata, outscale)


@core.register_scaled_lax_op
def scaled_cos(val: ScaledArray) -> ScaledArray:
    return scaled_op_default_translation(lax.cos_p, [val])


@core.register_scaled_lax_op
def scaled_sin(val: ScaledArray) -> ScaledArray:
    return scaled_op_default_translation(lax.sin_p, [val])


@core.register_scaled_lax_op
def scaled_integer_pow(A: ScaledArray, y: int) -> ScaledArray:
    output_scale = lax.integer_pow(A.scale, y)
    output_data = lax.integer_pow(A.data, y)
    return ScaledArray(output_data, output_scale)


@core.register_scaled_lax_op
def scaled_sqrt(val: ScaledArray) -> ScaledArray:
    output_scale = lax.sqrt(val.scale)
    output_data = lax.sqrt(val.data)
    return ScaledArray(output_data, output_scale)
