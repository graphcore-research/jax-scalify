# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import Any, Optional, Union

import jax
import jax.numpy as jnp
import numpy as np
from chex import Shape
from jax.core import ShapedArray
from jax.tree_util import register_pytree_node_class
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .typing import Array, ArrayTypes

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


def is_scaled_leaf(val: Any) -> bool:
    """Is input a JAX PyTree (scaled) leaf, including ScaledArray.

    This function is useful for JAX PyTree handling where the user wants
    to keep the ScaledArray datastructures (i.e. not flattened as a pair of arrays).
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


def as_scaled_array_base(val: Any, scale: Optional[ArrayLike] = None) -> Union[Array, ScaledArray]:
    """ScaledArray (helper) base factory method, similar to `(j)np.array`."""
    if isinstance(val, ScaledArray):
        return val

    # Simple case => when can ignore the scaling factor (i.e. 1 implicitely).
    is_static_one_scale: bool = scale is None or is_static_one_scalar(scale)  # type:ignore
    # Trivial cases: bool, int, float.
    if is_static_one_scale and isinstance(val, (bool, int)):
        return val
    if is_static_one_scale and isinstance(val, float):
        return ScaledArray(np.array(1, dtype=np.float32), np.float32(val))

    # Ignored dtypes by default: int and bool
    ignored_dtype = np.issubdtype(val.dtype, np.integer) or np.issubdtype(val.dtype, np.bool_)
    if ignored_dtype:
        return val
    # Floating point scalar
    if val.ndim == 0 and is_static_one_scale:
        return ScaledArray(np.array(1, dtype=val.dtype), val)

    scale = np.array(1, dtype=val.dtype) if scale is None else scale
    if isinstance(val, (np.ndarray, Array)):
        if is_static_one_scale:
            return ScaledArray(val, scale)
        else:
            return ScaledArray(val / scale.astype(val.dtype), scale)  # type:ignore
    return scaled_array_base(val, scale)


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
    return jax.tree_map(lambda x: as_scaled_array_base(x, scale), val, is_leaf=is_scaled_leaf)


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
    return jax.tree_map(lambda x: asarray_base(x, dtype), val, is_leaf=is_scaled_leaf)


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


def is_static_one_scalar(val: Array) -> Union[bool, np.bool_]:
    """Is a scaled array a static one scalar value (i.e. one during JAX tracing as well)?"""
    if isinstance(val, (int, float)):
        return val == 1
    elif is_numpy_scalar_or_array(val) and val.size == 1:
        return np.all(np.equal(val, 1))
    elif isinstance(val, ScaledArray) and val.size == 1:
        return is_static_one_scalar(val.data) and is_static_one_scalar(val.scale)
    return False
