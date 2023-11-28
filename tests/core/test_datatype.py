# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax.core import ShapedArray

from jax_scaled_arithmetics.core import Array, ScaledArray, as_scaled_array, asarray, is_scaled_leaf, scaled_array


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

    def test__is_scaled_leaf__consistent_with_jax(self):
        assert is_scaled_leaf(8)
        assert is_scaled_leaf(2.0)
        assert is_scaled_leaf(np.int32(2))
        assert is_scaled_leaf(np.float32(2))
        assert is_scaled_leaf(np.array(3))
        assert is_scaled_leaf(np.array([3]))
        assert is_scaled_leaf(jnp.array([3]))
        assert is_scaled_leaf(scaled_array(data=[1.0, 2.0], scale=3, dtype=np.float16))

    @parameterized.parameters(
        {"data": np.array(2)},
        {"data": np.array([1, 2])},
        {"data": jnp.array([1, 2])},
    )
    def test__as_scaled_array__unchanged_dtype(self, data):
        output = as_scaled_array(data)
        assert isinstance(output, ScaledArray)
        assert isinstance(output.data, type(data))
        assert output.dtype == data.dtype
        npt.assert_array_almost_equal(output, data)
        npt.assert_array_equal(output.scale, np.array(1, dtype=data.dtype))
        # unvariant when calling a second time.
        assert as_scaled_array(output) is output

    def test__as_scaled_array__complex_pytree(self):
        input = {"x": jnp.array([1, 2]), "y": as_scaled_array(jnp.array([1, 2]))}
        output = as_scaled_array(input)
        assert isinstance(output, dict)
        assert len(output) == 2
        assert isinstance(output["x"], ScaledArray)
        npt.assert_array_equal(output["x"], input["x"])
        assert output["y"] is input["y"]

    @parameterized.parameters(
        {"data": np.array(2)},
        {"data": np.array([1, 2])},
        {"data": jnp.array([1, 2])},
    )
    def test__asarray__unchanged_dtype(self, data):
        output = asarray(data)
        assert output.dtype == data.dtype
        npt.assert_array_almost_equal(output, data)

    @parameterized.parameters(
        {"data": np.array([1, 2])},
        {"data": jnp.array([1, 2])},
        {"data": scaled_array(data=[1.0, 2.0], scale=3, dtype=np.float32)},
    )
    def test__asarray__changed_dtype(self, data):
        output = asarray(data, dtype=np.float16)
        assert output.dtype == np.float16
        npt.assert_array_almost_equal(output, data)

    def test__asarray__complex_pytree(self):
        input = {"x": jnp.array([1.0, 2]), "y": scaled_array(jnp.array([3, 4.0]), jnp.array(0.5))}
        output = asarray(input)
        assert isinstance(output, dict)
        assert len(output) == 2
        assert all([isinstance(v, Array) for v in output.values()])
        npt.assert_array_almost_equal(output["x"], input["x"])
        npt.assert_array_almost_equal(output["y"], input["y"])
