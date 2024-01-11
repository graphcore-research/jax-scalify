# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax.core import ShapedArray

from jax_scaled_arithmetics.core import (
    Array,
    ScaledArray,
    as_scaled_array,
    asarray,
    get_scale_dtype,
    is_scaled_leaf,
    is_static_one_scalar,
    is_static_zero,
    make_scaled_scalar,
    pow2_round_down,
    scaled_array,
)


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
        {"dtype": np.float16},
        {"dtype": np.float32},
    )
    def test__scaled_array__factory_method__multi_dtypes(self, dtype):
        sarr = scaled_array(data=[1.0, 2.0], scale=3.0, dtype=dtype, npapi=jnp)
        assert isinstance(sarr, ScaledArray)
        assert isinstance(sarr.data, jnp.ndarray)
        assert isinstance(sarr.scale, jnp.ndarray)
        assert sarr.data.dtype == ShapedArray((2,), dtype=dtype)
        # Default scale dtype is always FP32.
        # TODO: what happens when integer scale is passed?
        assert sarr.scale.dtype == np.float32
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

    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__scaled_array__astype(self, npapi):
        arr_in = ScaledArray(data=npapi.array([1.0, 2.0], dtype=np.float16), scale=npapi.array(1, dtype=np.int32))
        arr_out = arr_in.astype(np.float32)
        assert arr_out.dtype == np.float32
        assert arr_out.scale.dtype == arr_in.scale.dtype

    @parameterized.parameters(
        {"val": 0.25},
    )
    def test__make_scaled_scalar__float_input(self, val):
        scaled_val = make_scaled_scalar(val)
        assert isinstance(scaled_val, ScaledArray)
        assert scaled_val.shape == ()
        assert scaled_val.data.dtype == np.float32
        assert scaled_val.scale.dtype == np.float32
        npt.assert_equal(np.asarray(scaled_val), val)
        assert isinstance(scaled_val.data, (np.ndarray, np.number))
        assert isinstance(scaled_val.scale, (np.ndarray, np.number))

    @parameterized.parameters(
        {"val": np.float16(0)},
        {"val": np.float32(0)},
        # {"val": np.float64(0)}, FIXME!
    )
    def test__make_scaled_scalar__zero_scalar_input(self, val):
        scaled_val = make_scaled_scalar(val)
        assert isinstance(scaled_val, ScaledArray)
        assert scaled_val.shape == ()
        assert scaled_val.dtype == val.dtype

    @parameterized.parameters(
        {"val": np.array(1.0)},
        {"val": np.float32(-0.5)},
        {"val": np.float16(1.25)},
        {"val": np.float32(-65504)},
        # Testing JAX arrays too.
        {"val": jnp.float32(0.5)},
        {"val": jnp.float16(1.25)},
    )
    def test__make_scaled_scalar__proper_split_data_scale(self, val):
        scaled_val = make_scaled_scalar(val)
        assert isinstance(scaled_val, ScaledArray)
        assert scaled_val.shape == ()
        assert scaled_val.data.dtype == val.dtype
        assert scaled_val.scale.dtype == val.dtype
        npt.assert_equal(np.asarray(scaled_val), val)
        npt.assert_equal(scaled_val.scale, pow2_round_down(val))
        npt.assert_array_less(0, scaled_val.scale)
        # Make sure we are not doing implicit conversion to JAX arrays.
        if isinstance(val, Array):
            assert isinstance(scaled_val.data, Array)
            assert isinstance(scaled_val.scale, Array)
        else:
            assert isinstance(scaled_val.data, (np.ndarray, np.number))
            assert isinstance(scaled_val.scale, (np.ndarray, np.number))

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
        {"data": np.array([1, 2.0])},
        {"data": jnp.array([1, 2.0])},
    )
    def test__as_scaled_array__unchanged_dtype(self, data):
        output = as_scaled_array(data)
        assert isinstance(output, ScaledArray)
        assert isinstance(output.data, type(data))
        assert output.dtype in {np.dtype(np.float32), np.dtype(np.float64)}
        npt.assert_array_almost_equal(output, data)
        npt.assert_array_equal(output.scale, np.array(1, dtype=data.dtype))
        # unvariant when calling a second time.
        assert as_scaled_array(output) is output

    @parameterized.parameters(
        {"data": 2.1},
        {"data": np.float64(2.0)},
        {"data": jnp.float32(2.0)},
    )
    def test__as_scaled_array__float_scalar(self, data):
        output = as_scaled_array(data)
        assert isinstance(output, ScaledArray)
        assert output.data.dtype == output.scale.dtype
        # NOTE: for scalars, data always in [1, 2)
        npt.assert_almost_equal(np.asarray(output), data)

    @parameterized.parameters(
        {"data": jnp.float32(3.0)},
    )
    def test__as_scaled_array__optional_scalar(self, data):
        scale = np.float16(2)
        output = as_scaled_array(data, scale=scale)
        assert isinstance(output, ScaledArray)
        assert output.scale.dtype == np.float16
        npt.assert_array_equal(output.scale, np.array(2, dtype=np.float16))
        npt.assert_array_almost_equal(output, data)

    @parameterized.parameters(
        {"data": False},
        {"data": 2},
        {"data": np.array([1, 2])},
        {"data": jnp.array([1, 2])},
    )
    def test__as_scaled_array__unscaled_bool_int_output(self, data):
        output = as_scaled_array(data)
        assert output is data

    def test__as_scaled_array__complex_pytree(self):
        input = {"x": jnp.array([1, 2]), "y": jnp.array([1.0, 2]), "z": as_scaled_array(jnp.array([1.0, 2]))}
        output = as_scaled_array(input)
        assert isinstance(output, dict)
        assert len(output) == 3
        assert output["x"] is input["x"]
        npt.assert_array_equal(output["y"], input["y"])
        assert output["z"] is input["z"]

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        {"data": np.int32(3)},
        {"data": np.array(2, dtype=np.int32)},
        {"data": np.array([1, 2], dtype=np.int32)},
        {"data": np.array([1, 2.0], dtype=np.float32)},
        {"data": jnp.array([1, 2])},
    )
    def test__asarray__unchanged_dtype(self, data):
        output = self.variant(asarray)(data)
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

    @parameterized.parameters(
        {"val": 0, "result": True},
        {"val": 0.0, "result": True},
        {"val": np.int32(0), "result": True},
        {"val": np.float16(0), "result": True},
        {"val": np.array([1, 2]), "result": False},
        {"val": np.array([0, 0.0]), "result": True},
        {"val": jnp.array([0, 0.0]), "result": False},
        {"val": ScaledArray(np.array([0, 0.0]), jnp.array(2.0)), "result": True},
        {"val": ScaledArray(jnp.array([3, 4.0]), np.array(0.0)), "result": True},
        {"val": ScaledArray(jnp.array([3, 4.0]), jnp.array(0.0)), "result": False},
    )
    def test__is_static_zero__proper_all_result(self, val, result):
        all_zero = np.all(is_static_zero(val))
        assert all_zero == result

    @parameterized.parameters(
        {"val": 0, "result": False},
        {"val": 1, "result": True},
        {"val": 1.0, "result": True},
        {"val": np.int32(1), "result": True},
        {"val": np.float16(1), "result": True},
        {"val": np.float32(1), "result": True},
        {"val": np.array([1]), "result": True},
        {"val": np.array([1]), "result": True},
        {"val": np.array([1, 1]), "result": False},
        {"val": jnp.array([1, 1]), "result": False},
        {"val": ScaledArray(np.array([1]), np.array(1.0)), "result": True},
        {"val": ScaledArray(np.array(1.0), np.array(1)), "result": True},
        {"val": ScaledArray(np.array([1]), np.array(2.0)), "result": False},
    )
    def test__is_static_one_scalar__proper_result(self, val, result):
        r = is_static_one_scalar(val)
        assert isinstance(r, (bool, np.bool_))
        assert r == result
        # Should still be valid when we use `asarray` as well.
        r = is_static_one_scalar(asarray(val))
        assert isinstance(r, (bool, np.bool_))
        assert r == result

    @parameterized.parameters(
        {"val": np.float32(2), "result": np.float32},
        {"val": ScaledArray(np.array(2.0), np.float16(3)), "result": np.float16},
    )
    def test__get_scale_dtype__proper_dtype(self, val, result):
        sdtype = get_scale_dtype(val)
        assert sdtype == result
