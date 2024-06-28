# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import Shape
from jax.core import ShapedArray
from jax.tree_util import register_pytree_node_class
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .pow2 import Pow2RoundMode, pow2_decompose
from .typing import Array, ArrayTypes

if TYPE_CHECKING:
    GenericArray = Union[Array, np.ndarray[Any, Any]]
else:
    GenericArray = Union[Array, np.ndarray]


@register_pytree_node_class
@dataclass
class ScaledArray:
    """ScaledArray: dataclass associating data and a scale.

    JAX Scaled Arithmetics provides a consistent JAX LAX implementation
    propagating scaling for low-precision arithmetics.

    Semantics: `ScaledArray` represents an array with the following values:
        self.data * self.scale
    where `self.scale` is always assumed to be broadcastable to `self.data`.

    Notes:
        1. Current implementation only supports `scale` being a scalar.
        2. `data` and `scale` can have different dtypes. `data` dtype is used as the
            reference dtype. Meaning a power of 2 `scale` is just a dtype `E8M0` for instance.

    Args:
        data: Un-scaled data array.
        scale: Scale array (scalar only supported at the moment).
            If `scale` is None, equivalent to a normal array.
    """

    data: GenericArray
    scale: GenericArray

    def __post_init__(self):
        # Always have a Numpy array as `data`.
        if isinstance(self.data, np.number):
            object.__setattr__(self, "data", np.array(self.data))
        # TODO/FIXME: support number as data?
        assert isinstance(self.data, (*ArrayTypes, np.ndarray))
        assert isinstance(self.scale, (*ArrayTypes, np.ndarray, np.number))
        # Only supporting scale scalar for now.
        assert self.scale.shape == ()

    def tree_flatten(self):
        # See official JAX documentation on extending PyTrees.
        # Note: using explicit tree flatten instead of chex for MyPy compatibility.
        children = (self.data, self.scale)
        return (children, None)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        # See official JAX documentation on extending PyTrees.
        assert len(children) == 2
        return cls(children[0], children[1])

    @property
    def dtype(self) -> DTypeLike:
        return self.data.dtype

    @property
    def shape(self) -> Shape:
        return self.data.shape

    @property
    def size(self) -> int:
        return self.data.size

    def to_array(self, dtype: DTypeLike = None) -> GenericArray:
        """Convert to the scaled array to a Numpy/JAX array.

        Args:
            dtype: Optional conversion dtype. `data.dtype` by default.
        """
        dtype = self.data.dtype if dtype is None else dtype
        data = self.data.astype(dtype)
        scale = self.scale.astype(dtype)
        values = data * scale
        return values

    def __array__(self, dtype: DTypeLike = None) -> NDArray[Any]:
        """Numpy array interface support."""
        return np.asarray(self.to_array(dtype))

    @property
    def aval(self) -> ShapedArray:
        """Abstract value of the scaled array, i.e. shape and dtype."""
        return ShapedArray(self.data.shape, self.data.dtype)

    def astype(self, dtype: DTypeLike) -> "ScaledArray":
        """Convert the ScaledArray to a dtype.
        NOTE: only impacting `data` field, not the `scale` tensor.
        """
        return ScaledArray(self.data.astype(dtype), self.scale)


def make_scaled_scalar(val: Array, scale_dtype: Optional[DTypeLike] = None) -> ScaledArray:
    """Make a scaled scalar (array), from a single value.

    The returned scalar will always be built such that:
        - data is scalar in [1, 2)
        - scale is a power-of-2 value.

    NOTE: data is chosen in [1, 2) instead of [0, 1) in order to
    keep any value representable in the same dtype, without overflowing.

    NOTE bis: only supporting floating point input.
    """
    # FIXME: implicit conversion from float64 to float32???
    if isinstance(val, float):
        val = np.float32(val)
    assert np.ndim(val) == 0
    assert np.issubdtype(val.dtype, np.floating)
    # Scale dtype to use. TODO: check the scale dtype is valid?
    scale_dtype = scale_dtype or val.dtype
    # Split mantissa and exponent in data and scale components.
    scale, mantissa = pow2_decompose(val, scale_dtype=scale_dtype, mode=Pow2RoundMode.DOWN)
    return ScaledArray(mantissa, scale)


def is_scaled_leaf(val: Any) -> bool:
    """Is input a normal JAX PyTree leaf (i.e. `Array`) or `ScaledArray1.

    This function is useful for JAX PyTree handling with `jax.tree` methods where
    the user wants to keep the ScaledArray data structures (i.e. not flattened as a
    pair of arrays).

    See `jax_scalify.tree` for PyTree `jax.tree` methods compatible with `ScaledArray`.
    """
    # TODO: check Numpy scalars as well?
    return np.isscalar(val) or isinstance(val, (Array, np.ndarray, ScaledArray))


def scaled_array_base(data: ArrayLike, scale: ArrayLike, dtype: DTypeLike = None, npapi: Any = jnp) -> ScaledArray:
    """ScaledArray (helper) base factory method, similar to `(j)np.array`."""
    data = npapi.asarray(data, dtype=dtype)
    scale = npapi.asarray(scale)
    return ScaledArray(data, scale)


def scaled_array(data: ArrayLike, scale: ArrayLike, dtype: DTypeLike = None, npapi: Any = jnp) -> ScaledArray:
    """ScaledArray (helper) factory method, similar to `(j)np.array`.

    Args:
        data: Main data/values.
        scale: Scale tensor.
        dtype: Optional dtype to use for the data.
        npapi: Numpy API to use.
    Returns:
        Scaled array instance.
    """
    return scaled_array_base(data, scale, dtype, npapi)


def as_scaled_array_base(
    val: Any, scale: Optional[ArrayLike] = None, scale_dtype: Optional[DTypeLike] = None
) -> Union[Array, ScaledArray]:
    """ScaledArray (helper) base factory method, similar to `(j)np.array`.

    Args:
        val: Value to convert to scaled array.
        scale: Optional scale value.
        scale_dtype: Optional (default) scale dtype.
    """
    if isinstance(val, ScaledArray):
        return val

    assert scale is None or scale_dtype is None
    # Simple case => when can ignore the scaling factor (i.e. 1 implicitely).
    is_static_one_scale: bool = scale is None or is_static_one_scalar(scale)  # type:ignore
    # Trivial cases: bool, int, float.
    if is_static_one_scale and isinstance(val, (bool, int)):
        return val
    if is_static_one_scale and isinstance(val, float):
        return make_scaled_scalar(np.float32(val), scale_dtype)

    # Ignored dtypes by default: int and bool
    ignored_dtype = np.issubdtype(val.dtype, np.integer) or np.issubdtype(val.dtype, np.bool_)
    if ignored_dtype:
        return val
    # Floating point scalar
    if val.ndim == 0 and is_static_one_scale:
        return make_scaled_scalar(val, scale_dtype)

    scale_dtype = scale_dtype or val.dtype
    scale = np.array(1, dtype=scale_dtype) if scale is None else scale
    if isinstance(val, (np.ndarray, *ArrayTypes)):
        if is_static_one_scale:
            return ScaledArray(val, scale)
        else:
            return ScaledArray(val / scale.astype(val.dtype), scale)  # type:ignore

    # TODO: fix bug when scale is not 1.
    raise NotImplementedError(f"Constructing `ScaledArray` from {val} and {scale} not supported.")
    # return scaled_array_base(val, scale)


def as_scaled_array(val: Any, scale: Optional[ArrayLike] = None) -> ScaledArray:
    """ScaledArray (helper) factory method, similar to `(j)np.array`.

    NOTE: by default, int and bool values/arrays will be returned unchanged, as
    in most cases, there is no value representing these as scaled arrays.

    Compatible with JAX PyTree.

    Args:
        val: Main data/values or existing ScaledArray.
        scale: Optional scale to use when (potentially) converting.
    Returns:
        Scaled array instance.
    """
    return jax.tree_util.tree_map(lambda x: as_scaled_array_base(x, scale), val, is_leaf=is_scaled_leaf)


def asarray_base(val: Any, dtype: DTypeLike = None) -> GenericArray:
    """Convert back to a common JAX/Numpy array, base function."""
    if isinstance(val, ScaledArray):
        return val.to_array(dtype=dtype)
    elif isinstance(val, (*ArrayTypes, np.ndarray)):
        if dtype is None:
            return val
        return val.astype(dtype=dtype)
    # Convert to Numpy all other cases?
    return np.asarray(val, dtype=dtype)


def asarray(val: Any, dtype: DTypeLike = None) -> GenericArray:
    """Convert back to a common JAX/Numpy array.

    Compatible with JAX PyTree.

    Args:
        dtype: Optional dtype of the final array.
    """
    return jax.tree_util.tree_map(lambda x: asarray_base(x, dtype), val, is_leaf=is_scaled_leaf)


def is_numpy_scalar_or_array(val):
    return isinstance(val, np.ndarray) or np.isscalar(val)


def is_static_zero(val: Union[Array, ScaledArray]) -> Array:
    """Is a scaled array a static zero value (i.e. zero during JAX tracing as well)?

    Returns a boolean Numpy array of the shape of the input.
    """
    if is_numpy_scalar_or_array(val):
        return np.equal(val, 0)
    if isinstance(val, ScaledArray):
        data_mask = (
            np.equal(val.data, 0) if is_numpy_scalar_or_array(val.data) else np.zeros(val.data.shape, dtype=np.bool_)
        )
        scale_mask = (
            np.equal(val.scale, 0) if is_numpy_scalar_or_array(val.scale) else np.zeros(val.scale.shape, dtype=np.bool_)
        )
        return np.logical_or(data_mask, scale_mask)
    # By default: can't decide.
    return np.zeros(val.shape, dtype=np.bool_)


def is_static_anyscale(val: Union[Array, ScaledArray]) -> Array:
    """Is a scaled array a static anyscale values (i.e. 0/inf/-inf during JAX tracing as well)?

    Returns a boolean Numpy array of the shape of the input.
    """

    def np_anyscale(arr):
        # Check if 0, np.inf or -np.inf
        absarr = np.abs(arr)
        return np.logical_or(np.equal(absarr, 0), np.equal(absarr, np.inf))

    if is_numpy_scalar_or_array(val):
        return np_anyscale(val)
    if isinstance(val, ScaledArray):
        # TODO: deal with 0 * inf issue?
        data_mask = (
            np_anyscale(val.data) if is_numpy_scalar_or_array(val.data) else np.zeros(val.data.shape, dtype=np.bool_)
        )
        scale_mask = (
            np_anyscale(val.scale) if is_numpy_scalar_or_array(val.scale) else np.zeros(val.scale.shape, dtype=np.bool_)
        )
        return np.logical_or(data_mask, scale_mask)
    # By default: can't decide.
    return np.zeros(val.shape, dtype=np.bool_)


def is_static_one_scalar(val: Array) -> Union[bool, np.bool_]:
    """Is a scaled array a static one scalar value (i.e. one during JAX tracing as well)?"""
    if isinstance(val, (int, float)):
        return val == 1
    elif is_numpy_scalar_or_array(val) and val.size == 1:
        return np.all(np.equal(val, 1))
    elif isinstance(val, ScaledArray) and val.size == 1:
        return is_static_one_scalar(val.data) and is_static_one_scalar(val.scale)
    return False


def get_scale_dtype(val: Any) -> DTypeLike:
    """Get the scale dtype. Compatible with arrays and scaled arrays."""
    if isinstance(val, ScaledArray):
        return val.scale.dtype
    return val.dtype
