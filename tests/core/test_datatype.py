# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scaled_arithmetics.core import ScaledArray


class ScaledArrayDataclassTests(chex.TestCase):
    @parameterized.parameters(
        {"npb": np},
        {"npb": jnp},
    )
    def test__scaled_array__init__multi_numpy_backend(self, npb):
        sarr = ScaledArray(data=npb.array([1.0, 2.0], dtype=np.float32), scale=npb.array(1))
        assert isinstance(sarr.data, npb.ndarray)
        assert isinstance(sarr.scale, npb.ndarray)
        assert sarr.scale.shape == ()

    def test__scaled_array__basic_properties(self):
        sarr = ScaledArray(data=jnp.array([1.0, 2.0]), scale=jnp.array(1))
        assert sarr.dtype == np.float32
        assert sarr.shape == (2,)

    @parameterized.parameters(
        {"npb": np},
        {"npb": jnp},
    )
    def test__scaled_array__to_array__multi_numpy_backend(self, npb):
        sarr = ScaledArray(data=npb.array([1.0, 2.0], dtype=np.float16), scale=npb.array(3))
        # No dtype specified.
        out = sarr.to_array()
        assert isinstance(out, npb.ndarray)
        assert out.dtype == sarr.dtype
        npt.assert_array_equal(out, sarr.data * sarr.scale)
        # Custom float dtype.
        out = sarr.to_array(dtype=np.float32)
        assert isinstance(out, npb.ndarray)
        assert out.dtype == np.float32
        npt.assert_array_equal(out, sarr.data * sarr.scale)
        # Custom int dtype.
        out = sarr.to_array(dtype=np.int8)
        assert isinstance(out, npb.ndarray)
        assert out.dtype == np.int8
        npt.assert_array_equal(out, sarr.data * sarr.scale)

    @parameterized.parameters(
        {"npb": np},
        {"npb": jnp},
    )
    def test__scaled_array__numpy_array_interface(self, npb):
        sarr = ScaledArray(data=npb.array([1.0, 2.0], dtype=np.float32), scale=npb.array(3))
        out = np.asarray(sarr)
        assert isinstance(out, np.ndarray)
        npt.assert_array_equal(out, sarr.data * sarr.scale)
