# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scaled_arithmetics.core import ScaledArray, autoscale, scaled_array
from jax_scaled_arithmetics.lax import set_scaling, stop_scaling


class SetScalingPrimitiveTests(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    def test__set_scaling_primitive__proper_result_without_autoscale(self):
        def fn(arr, scale):
            return set_scaling(arr, scale)

        fn = self.variant(fn)
        arr = jnp.array([2, 3], dtype=np.float32)
        scale = jnp.array(4, dtype=np.float32)
        out = fn(arr, scale)
        npt.assert_array_equal(out, arr)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        # Testing different combination of scaled/unscaled inputs.
        {"arr": np.array([-1.0, 2.0], dtype=np.float32), "scale": np.array(4.0, dtype=np.float32)},
        {"arr": scaled_array([-1.0, 2.0], 1.0, dtype=np.float32), "scale": np.array(4.0, dtype=np.float32)},
        {"arr": scaled_array([-1.0, 2.0], 1.0, dtype=np.float32), "scale": scaled_array(1.0, 4.0, dtype=np.float32)},
    )
    def test__set_scaling_primitive__proper_result_with_autoscale(self, arr, scale):
        def fn(arr, scale):
            return set_scaling(arr, scale)

        fn = self.variant(autoscale(fn))
        out = fn(arr, scale)
        # Unchanged output tensor!
        assert isinstance(out, ScaledArray)
        npt.assert_array_equal(out.scale, scale)
        npt.assert_array_equal(out, arr)


class StopScalingPrimitiveTests(chex.TestCase):
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
        assert isinstance(out0, jax.Array)
        assert isinstance(out1, jax.Array)
        assert out0.dtype == arr.dtype
        assert out1.dtype == np.float16
        npt.assert_array_equal(out0, arr)
        npt.assert_array_almost_equal(out1, arr)
