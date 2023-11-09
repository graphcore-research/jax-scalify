# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax.core import ShapedArray

from jax_scaled_arithmetics import ScaledArray, scaled_array


class ScaledArrayDataclassTests(chex.TestCase):
    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__scaled_array__init__multi_numpy_backend(self, npapi):
        sarr = ScaledArray(data=npapi.array([1.0, 2.0], dtype=np.float32), scale=npapi.array(1))
        assert isinstance(sarr.data, npapi.ndarray)
        assert isinstance(sarr.scale, npapi.ndarray)
        assert sarr.scale.shape == ()

    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__scaled_array__factory_method__multi_numpy_backend(self, npapi):
        sarr = scaled_array(data=[1.0, 2.0], scale=3, dtype=np.float16, npapi=npapi)
        assert isinstance(sarr, ScaledArray)
        assert isinstance(sarr.data, npapi.ndarray)
        assert isinstance(sarr.scale, npapi.ndarray)
        assert sarr.data.dtype == ShapedArray((2,), np.float16)
        assert sarr.scale.shape == ()
        npt.assert_array_almost_equal(sarr, [3, 6])

    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__scaled_array__basic_properties(self, npapi):
        sarr = ScaledArray(data=npapi.array([1.0, 2.0], dtype=np.float32), scale=npapi.array(1))
        assert sarr.dtype == np.float32
        assert sarr.shape == (2,)
        assert sarr.aval == ShapedArray((2,), np.float32)

    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__scaled_array__to_array__multi_numpy_backend(self, npapi):
        sarr = scaled_array(data=[1.0, 2.0], scale=3, dtype=np.float16, npapi=npapi)
        # No dtype specified.
        out = sarr.to_array()
        assert isinstance(out, npapi.ndarray)
        assert out.dtype == sarr.dtype
        npt.assert_array_equal(out, sarr.data * sarr.scale)
        # Custom float dtype.
        out = sarr.to_array(dtype=np.float32)
        assert isinstance(out, npapi.ndarray)
        assert out.dtype == np.float32
        npt.assert_array_equal(out, sarr.data * sarr.scale)
        # Custom int dtype.
        out = sarr.to_array(dtype=np.int8)
        assert isinstance(out, npapi.ndarray)
        assert out.dtype == np.int8
        npt.assert_array_equal(out, sarr.data * sarr.scale)

    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__scaled_array__numpy_array_interface(self, npapi):
        sarr = ScaledArray(data=npapi.array([1.0, 2.0], dtype=np.float32), scale=npapi.array(3))
        out = np.asarray(sarr)
        assert isinstance(out, np.ndarray)
        npt.assert_array_equal(out, sarr.data * sarr.scale)
