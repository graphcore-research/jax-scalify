# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax
import jax.numpy as jnp
import numpy as np

from jax_scaled_arithmetics.core import ScaledArray


class IpuTileArrayPrimitiveTests(chex.TestCase):
    def test__scaled_array__init__jax(self):
        sarr = ScaledArray(data=jnp.array([1.0, 2.0]), scale=jnp.array(1))
        assert isinstance(sarr.data, jax.Array)
        assert isinstance(sarr.scale, jax.Array)
        assert sarr.scale.shape == ()

    def test__scaled_array__init__numpy(self):
        sarr = ScaledArray(data=np.array([1.0, 2.0]), scale=np.array(1))
        assert isinstance(sarr.data, np.ndarray)
        assert isinstance(sarr.scale, np.ndarray)
        assert sarr.scale.shape == ()

    def test__scaled_array__basic_properties(self):
        sarr = ScaledArray(data=jnp.array([1.0, 2.0]), scale=jnp.array(1))
        assert sarr.dtype == np.float32
        assert sarr.shape == (2,)
