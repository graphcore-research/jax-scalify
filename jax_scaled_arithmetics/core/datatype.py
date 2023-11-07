# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import Any, Union

import jax
import numpy as np
from chex import Shape
from jax.tree_util import register_pytree_node_class
from numpy.typing import DTypeLike, NDArray

GenericArray = Union[jax.Array, np.ndarray]


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
        assert isinstance(self.data, (jax.Array, np.ndarray))
        assert isinstance(self.scale, (jax.Array, np.ndarray))
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
