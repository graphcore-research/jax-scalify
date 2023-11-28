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


def as_scaled_array_base(val: Any, scale: Optional[ArrayLike] = None) -> ScaledArray:
    """ScaledArray (helper) base factory method, similar to `(j)np.array`."""
    scale = np.array(1, dtype=val.dtype) if scale is None else scale
    if isinstance(val, ScaledArray):
        return val
    elif isinstance(val, (np.ndarray, Array)):
        return ScaledArray(val, scale)
    return scaled_array_base(val, scale)


def as_scaled_array(val: Any, scale: Optional[ArrayLike] = None) -> ScaledArray:
    """ScaledArray (helper) factory method, similar to `(j)np.array`.

    Compatible with JAX PyTree.

    Args:
        val: Main data/values or existing ScaledArray.
        scale: Optional scale to use when (potentially) converting.
    Returns:
        Scaled array instance.
    """
    return jax.tree_map(lambda x: as_scaled_array_base(x, None), val, is_leaf=is_scaled_leaf)


def asarray_base(val: Any, dtype: DTypeLike = None) -> GenericArray:
    """Convert back to a common JAX/Numpy array, base function."""
    if isinstance(val, ScaledArray):
        return val.to_array(dtype=dtype)
    elif isinstance(val, (Array, np.ndarray)):
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
