# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Optional, Sequence, Union

import numpy as np
from jax import core
from jax.interpreters import mlir
from jax.interpreters.mlir import LoweringRuleContext, ir, ir_constant

from jax_scaled_arithmetics.core import (
    Array,
    DTypeLike,
    ScaledArray,
    ScaledPrimitiveType,
    asarray,
    is_static_one_scalar,
    register_scaled_op,
)

set_scaling_p = core.Primitive("set_scaling_p")
"""`set_scaling` JAX primitive.

In standard JAX, this is just an identity operation, ignoring the `scale`
input, just returning unchanged the `data` component.

In JAX Scaled Arithmetics/AutoScale mode, it will rebalance the data term to
return a ScaledArray semantically equivalent.
"""


def set_scaling(values: Array, scale: Array) -> Array:
    """`set_scaling` primitive call method."""
    return set_scaling_p.bind(values, scale)


def set_scaling_impl(values: Array, scale: Array) -> Array:
    return values


def set_scaling_abstract_eval(values: core.ShapedArray, scale: core.ShapedArray) -> core.ShapedArray:
    return values


def set_scaling_mlir_lowering(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]]
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    # Just forwarding `values` term, ignoring the `scale`.
    return (args[0],)


def scaled_set_scaling(values: ScaledArray, scale: ScaledArray) -> ScaledArray:
    """Scaled `set_scaling` implementation: rebalancing the data using the new scale value."""
    # Trivial case of scale == 1
    if is_static_one_scalar(scale):
        if isinstance(values, ScaledArray):
            return values
        return ScaledArray(values, scale)
    assert scale.shape == ()
    # Automatic promotion should ensure we always get a scaled scalar here!
    scale_value = asarray(scale)
    if not isinstance(values, ScaledArray):
        # Simple case, with no pre-existing scale.
        return ScaledArray(values / scale_value.astype(values.dtype), scale_value)
    # Rebalancing data tensor using the new scale.
    data = values.data * (values.scale / scale_value).astype(values.dtype)
    return ScaledArray(data, scale_value)


# Register as standard JAX primitive
set_scaling_p.multiple_results = False
set_scaling_p.def_abstract_eval(set_scaling_abstract_eval)
set_scaling_p.def_impl(set_scaling_impl)
mlir.register_lowering(set_scaling_p, set_scaling_mlir_lowering)
# Register "scaled" translation.
register_scaled_op(set_scaling_p, scaled_set_scaling, ScaledPrimitiveType.ALWAYS_SCALE)


stop_scaling_p = core.Primitive("stop_scaling_p")
"""`stop_scaling` JAX primitive.

In standard JAX, this is just an identity operation (with optional casting).

In JAX Scaled Arithmetics/AutoScale mode, it will return the value tensor,
with optional casting.

Similar in principle to `jax.lax.stop_gradient`
"""


def stop_scaling(values: Array, dtype: Optional[DTypeLike] = None) -> Array:
    """`stop_scaling` primitive call method."""
    return stop_scaling_p.bind(values, dtype=dtype)


def stop_scaling_impl(values: Array, dtype: Optional[DTypeLike]) -> Array:
    if dtype is not None:
        values = values.astype(dtype)
    return values


def stop_scaling_abstract_eval(values: core.ShapedArray, dtype: Optional[DTypeLike]) -> core.ShapedArray:
    return values.update(dtype=dtype)


def stop_scaling_mlir_lowering(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]], **params
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    dtype = params.get("dtype", None)
    if dtype is not None:
        # TODO: caching of the MLIR lowered function?
        stop_scaling_mlir_fn = mlir.lower_fun(lambda x: x.astype(dtype), multiple_results=False)
        return stop_scaling_mlir_fn(ctx, *args)
    # By default: forward tensor.
    return (args[0],)


def scaled_stop_scaling(values: ScaledArray, dtype: Optional[DTypeLike] = None) -> Array:
    """Scaled `stop_scaling` implementation: returning tensor values (with optional cast)."""
    assert isinstance(values, ScaledArray)
    # TODO/FIXME: how to handle not scaled input?
    return values.to_array(dtype=dtype)


# Register as standard JAX primitive
stop_scaling_p.multiple_results = False
stop_scaling_p.def_abstract_eval(stop_scaling_abstract_eval)
stop_scaling_p.def_impl(stop_scaling_impl)
mlir.register_lowering(stop_scaling_p, stop_scaling_mlir_lowering)
# Register "scaled" translation.
register_scaled_op(stop_scaling_p, scaled_stop_scaling)


get_data_scale_p = core.Primitive("get_data_scale_p")
"""`get_data_scale` unbundling JAX primitive: return a tuple of data and scale
arrays.

In standard JAX, this is just an operation returning the input array and a constant scalar(1).

In JAX Scaled Arithmetics/AutoScale mode, it will return the pair of data and scale tensors
from a ScaledArray.
"""


def get_data_scale(values: Array) -> Array:
    """`get_data_scale` primitive call method."""
    return get_data_scale_p.bind(values)


def get_data_scale_impl(values: Array) -> Array:
    scale = np.ones((), dtype=values.dtype)
    return values, scale


def get_data_scale_abstract_eval(values: core.ShapedArray) -> core.ShapedArray:
    return values, core.ShapedArray((), dtype=values.dtype)


def get_data_scale_mlir_lowering(
    ctx: LoweringRuleContext, *args: Union[ir.Value, Sequence[ir.Value]]
) -> Sequence[Union[ir.Value, Sequence[ir.Value]]]:
    # Just forwarding `values` term, adding a constant scalar scale(1).
    assert len(args) == 1
    scale = ir_constant(np.ones((), dtype=ctx.avals_in[0].dtype))
    return (args[0], scale)


def scaled_get_data_scale(values: ScaledArray) -> Array:
    """Scaled `get_data_scale` implementation: return scale tensor."""
    return values.data, values.scale


# Register as standard JAX primitive
get_data_scale_p.multiple_results = True
get_data_scale_p.def_abstract_eval(get_data_scale_abstract_eval)
get_data_scale_p.def_impl(get_data_scale_impl)
mlir.register_lowering(get_data_scale_p, get_data_scale_mlir_lowering)
# Register "scaled" translation.
register_scaled_op(get_data_scale_p, scaled_get_data_scale)
