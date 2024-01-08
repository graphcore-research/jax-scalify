# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scaled_arithmetics.core import Array, ScaledArray, autoscale, scaled_array
from jax_scaled_arithmetics.lax.base_scaling_primitives import (
    get_data_scale,
    rebalance,
    scaled_set_scaling,
    set_scaling,
    stop_scaling,
)


class SetScalingPrimitiveTests(chex.TestCase):
    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__set_scaling_primitive__scaled_array__eager_mode(self, npapi):
        values = scaled_array([-1.0, 2.0], 2.0, dtype=np.float16, npapi=npapi)
        scale = npapi.float16(4)
        output = set_scaling(values, scale)
        assert isinstance(output, ScaledArray)
        # No implicit promotion to JAX array when not necessary.
        assert isinstance(output.data, npapi.ndarray)
        assert isinstance(output.scale, (npapi.number, npapi.ndarray))
        npt.assert_equal(output.scale, npapi.float16(4))
        npt.assert_array_equal(output, values)

    @chex.variants(with_jit=True, without_jit=True)
    def test__set_scaling_primitive__proper_result_without_autoscale(self):
        def fn(arr, scale):
            return set_scaling(arr, scale)

        fn = self.variant(fn)
        arr = jnp.array([2, 3], dtype=np.float32)
        scale = jnp.array(4, dtype=np.float32)
        out = fn(arr, scale)
        # Scale input ignored => forward same array.
        assert isinstance(out, jnp.ndarray)
        npt.assert_array_equal(out, arr)

    @chex.variants(with_jit=True, without_jit=False)
    @parameterized.parameters(
        # Testing different combination of scaled/unscaled inputs.
        {"arr": np.array([-1.0, 2.0], dtype=np.float32), "scale": np.array(4.0, dtype=np.float32)},
        {"arr": np.array([-1.0, 2.0], dtype=np.float16), "scale": np.array(4.0, dtype=np.float32)},
        {"arr": scaled_array([-1.0, 2.0], 1.0, dtype=np.float16), "scale": np.array(4.0, dtype=np.float32)},
        {"arr": scaled_array([-1.0, 2.0], 2.0, dtype=np.float32), "scale": scaled_array(1.0, 4.0, dtype=np.float32)},
        {"arr": scaled_array([-1.0, 2.0], 2.0, dtype=np.float16), "scale": scaled_array(1.0, 4.0, dtype=np.float32)},
    )
    def test__set_scaling_primitive__proper_result_with_autoscale(self, arr, scale):
        def fn(arr, scale):
            return set_scaling(arr, scale)

        fn = self.variant(autoscale(fn))
        out = fn(arr, scale)
        # Unchanged output tensor, with proper dtype.
        assert isinstance(out, ScaledArray)
        assert out.dtype == arr.dtype
        npt.assert_array_equal(out.scale, scale)
        npt.assert_array_equal(out, arr)

    @parameterized.parameters(
        {"scale": 1},
        {"scale": np.int32(1)},
        {"scale": np.float32(1)},
    )
    def test__scaled_set_scaling__unchanged_scaled_array(self, scale):
        val = scaled_array([-1.0, 2.0], 2.0, dtype=np.float16)
        assert scaled_set_scaling(val, scale) is val

    @parameterized.parameters(
        {"scale": np.int32(1)},
        {"scale": np.float32(1)},
    )
    def test__scaled_set_scaling__unchanged_data_scaled_array(self, scale):
        val = np.array([-1.0, 2.0], dtype=np.float16)
        out = scaled_set_scaling(val, scale)  # type:ignore
        assert isinstance(out, ScaledArray)
        assert out.data is val


class StopScalingPrimitiveTests(chex.TestCase):
    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__stop_scaling_primitive__scaled_array__eager_mode(self, npapi):
        values = scaled_array([-1.0, 2.0], 2.0, dtype=np.float16, npapi=npapi)
        output = stop_scaling(values)
        assert isinstance(output, npapi.ndarray)
        npt.assert_array_equal(output, values)

    @chex.variants(with_jit=True, without_jit=True)
    def test__stop_scaling_primitive__proper_result_without_autoscale(self):
        def fn(arr):
            # Testing both variants.
            return stop_scaling(arr), stop_scaling(arr, dtype=np.float16)

        arr = jnp.array([2, 3], dtype=np.float32)
        out0, out1 = self.variant(fn)(arr)
        assert out0.dtype == arr.dtype
        assert out1.dtype == np.float16
        npt.assert_array_equal(out0, arr)
        npt.assert_array_almost_equal(out1, arr)

    @chex.variants(with_jit=True, without_jit=True)
    def test__stop_scaling_primitive__proper_result_with_autoscale(self):
        def fn(arr):
            # Testing both variants.
            return stop_scaling(arr), stop_scaling(arr, dtype=np.float16)

        fn = self.variant(autoscale(fn))
        arr = scaled_array([-1.0, 2.0], 3.0, dtype=np.float32)
        out0, out1 = fn(arr)
        assert isinstance(out0, Array)
        assert isinstance(out1, Array)
        assert out0.dtype == arr.dtype
        assert out1.dtype == np.float16
        npt.assert_array_equal(out0, arr)
        npt.assert_array_almost_equal(out1, arr)


class GetDataScalePrimitiveTests(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test__get_data_scale_primitive__proper_result_without_autoscale(self):
        def fn(arr):
            return get_data_scale(arr)

        fn = self.variant(fn)
        arr = jnp.array([2, 3], dtype=np.float16)
        data, scale = fn(arr)
        npt.assert_array_equal(data, arr)
        npt.assert_equal(scale, np.array(1, arr.dtype))

    @chex.variants(with_jit=True, without_jit=True)
    def test__get_data_scale_primitive__proper_result_with_autoscale(self):
        def fn(arr):
            return get_data_scale(arr)

        fn = self.variant(autoscale(fn))
        arr = scaled_array([2, 3], 4, dtype=np.float16)
        data, scale = fn(arr)
        npt.assert_array_equal(data, arr.data)
        npt.assert_equal(scale, arr.scale)

    def test__get_data_scale_primitive__numpy_input(self):
        arr = scaled_array([2, 3], 4, dtype=np.float16)
        # ScaledArray input.
        data, scale = get_data_scale(arr)
        npt.assert_array_equal(data, arr.data)
        npt.assert_array_equal(scale, arr.scale)
        # Normal numpy array input.
        data, scale = get_data_scale(np.asarray(arr))
        npt.assert_array_equal(data, arr)
        npt.assert_almost_equal(scale, 1)


class RebalancingOpsTests(chex.TestCase):
    @parameterized.parameters(
        {"npapi": np},
        {"npapi": jnp},
    )
    def test__rebalance_op__normal_array__eagermode(self, npapi):
        values = npapi.array([-1, 2], dtype=np.float16)
        output = rebalance(values, np.float16(2))
        # Same Python array object forwarded.
        assert output is values

    def test__rebalance_op__scaled_array__eagermode(self):
        arr_in = scaled_array([2, 3], np.float16(4), dtype=np.float16)
        scale_in = np.float16(0.5)
        arr_out = rebalance(arr_in, scale_in)
        assert isinstance(arr_out, ScaledArray)
        npt.assert_array_equal(arr_out.scale, arr_in.scale * scale_in)
        npt.assert_array_equal(arr_out, arr_in)
