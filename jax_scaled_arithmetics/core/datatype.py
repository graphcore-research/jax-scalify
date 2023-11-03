# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from dataclasses import dataclass
from typing import Optional, Union

import jax
import numpy as np
from chex import Shape
from jax.tree_util import register_pytree_node_class
from numpy.typing import DTypeLike

GenericArray = Union[jax.Array, np.ndarray]


@register_pytree_node_class
@dataclass
class ScaledArray:
    """ScaledArray: dataclass associating data and a scale.

    JAX Scaled Arithmetics provides a consistent JAX LAX implementation
    propagating scaling for low-precision arithmetics.

    Args:
        data: Un-scaled data array.
        scale: Scale array (scalar only supported).
            If `scale` is None, equivalent to a normal array.
    """

    data: GenericArray
    scale: Optional[GenericArray] = None

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
