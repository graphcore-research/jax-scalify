# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax import lax

from jax_scaled_arithmetics.core import (
    Array,
    ScaledArray,
    autoscale,
    debug_callback,
    find_registered_scaled_op,
    scaled_array,
)
from jax_scaled_arithmetics.lax import (
    scaled_broadcast_in_dim,
    scaled_concatenate,
    scaled_convert_element_type,
    scaled_is_finite,
    scaled_pad,
    scaled_reduce_precision,
    scaled_reshape,
    scaled_rev,
    scaled_select_n,
    scaled_slice,
    scaled_transpose,
)


class ScaledTranslationPrimitivesTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    @chex.variants(with_jit=True, without_jit=True)
    def test__scaled_debug_callback__proper_forwarding(self):
        host_values = []

        def callback(*args):
            for v in args:
                host_values.append(v)

        def fn(a):
            # NOTE: multiplying by a power of 2 to simplify test.
            debug_callback(callback, a, a * 4)
            return a

        x = scaled_array(self.rs.rand(5), 2, dtype=np.float16)
        fn = self.variant(autoscale(fn))
        fn(x)

        assert len(host_values) == 2
        for sv in host_values:
            assert isinstance(sv, ScaledArray)
            npt.assert_array_equal(sv.data, x.data)
        npt.assert_array_equal(host_values[0].scale, x.scale)
        npt.assert_array_equal(host_values[1].scale, x.scale * 4)

    def test__scaled_broadcast_in_dim__proper_scaling(self):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
        z = scaled_broadcast_in_dim(x, shape=(5, 1), broadcast_dimensions=(0,))
        assert isinstance(z, ScaledArray)
        npt.assert_array_equal(z.scale, x.scale)
        npt.assert_array_almost_equal(z.data, x.data.reshape((5, 1)))

    def test__scaled_reshape__proper_scaling(self):
        x = scaled_array(self.rs.rand(8), 2, dtype=np.float32)
        z = scaled_reshape(x, new_sizes=(4, 2), dimensions=None)
        assert isinstance(z, ScaledArray)
        npt.assert_array_equal(z.scale, x.scale)
        npt.assert_array_almost_equal(z.data, x.data.reshape((4, 2)))

    def test__scaled_concatenate__proper_scaling(self):
        x = scaled_array(self.rs.rand(2, 3), 0.5, dtype=np.float16)
        y = scaled_array(self.rs.rand(5, 3), 2, dtype=np.float16)
        z = scaled_concatenate([x, y], dimension=0)
        assert isinstance(z, ScaledArray)
        assert z.dtype == x.dtype
        npt.assert_array_equal(z.scale, y.scale)
        npt.assert_array_almost_equal(z, lax.concatenate([np.asarray(x), np.asarray(y)], dimension=0))

    def test__scaled_concatenate__zero_input_scales(self):
        x = scaled_array(self.rs.rand(2, 3), 0.0, dtype=np.float16)
        y = scaled_array(self.rs.rand(5, 3), 0.0, dtype=np.float16)
        z = scaled_concatenate([x, y], dimension=0)
        assert isinstance(z, ScaledArray)
        assert z.dtype == x.dtype
        npt.assert_array_equal(z.scale, 0)
        npt.assert_array_almost_equal(z, lax.concatenate([np.asarray(x), np.asarray(y)], dimension=0))

    def test__scaled_convert_element_type__proper_scaling(self):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
        z = scaled_convert_element_type(x, new_dtype=np.float16)
        assert isinstance(z, ScaledArray)
        npt.assert_array_equal(z.scale, x.scale)
        npt.assert_array_almost_equal(z.data, x.data.astype(z.dtype))

    def test__scaled_transpose__proper_scaling(self):
        x = scaled_array(self.rs.rand(3, 5), 2, dtype=np.float32)
        z = scaled_transpose(x, (1, 0))
        assert isinstance(z, ScaledArray)
        assert z.scale == x.scale
        npt.assert_array_almost_equal(z.data, x.data.T)

    def test__scaled_rev__proper_scaling(self):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
        z = scaled_rev(x, dimensions=(0,))
        assert isinstance(z, ScaledArray)
        assert z.scale == x.scale
        npt.assert_array_almost_equal(z.data, x.data[::-1])

    def test__scaled_pad__proper_scaling(self):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
        z = scaled_pad(x, 0.0, [(1, 2, 0)])
        assert isinstance(z, ScaledArray)
        assert z.scale == x.scale
        npt.assert_array_almost_equal(z.data, lax.pad(x.data, 0.0, [(1, 2, 0)]))

    def test__scaled_reduce_precision__proper_result(self):
        x = scaled_array(self.rs.rand(3, 5), 2, dtype=np.float16)
        # Reduction to pseudo FP8 format.
        z = scaled_reduce_precision(x, exponent_bits=4, mantissa_bits=3)
        assert isinstance(z, ScaledArray)
        assert z.dtype == x.dtype
        assert z.scale == x.scale
        npt.assert_array_almost_equal(z.data, lax.reduce_precision(x.data, exponent_bits=4, mantissa_bits=3))

    def test__scaled_slice__proper_scaling(self):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
        z = scaled_slice(x, (1,), (4,), (2,))
        assert isinstance(z, ScaledArray)
        assert z.scale == x.scale
        npt.assert_array_almost_equal(z.data, x.data[1:4:2])

    @parameterized.parameters({"prim": lax.argmax_p}, {"prim": lax.argmin_p})
    def test__scaled_argminmax__proper_scaling(self, prim):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
        expected_out = prim.bind(x.to_array(), axes=(0,), index_dtype=np.int32)
        scaled_translation, _ = find_registered_scaled_op(prim)
        out = scaled_translation(x, axes=(0,), index_dtype=np.int32)
        assert isinstance(out, Array)
        npt.assert_array_equal(out, expected_out)


class ScaledTranslationBooleanPrimitivesTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    @parameterized.parameters(
        {"val": scaled_array([2, 3], 2.0, dtype=np.float32), "expected_out": [True, True]},
        # Supporting `int` scale as well.
        {"val": scaled_array([2, np.inf], 2, dtype=np.float32), "expected_out": [True, False]},
        {"val": scaled_array([2, 3], np.nan, dtype=np.float32), "expected_out": [False, False]},
        {"val": scaled_array([np.nan, 3], 3.0, dtype=np.float32), "expected_out": [False, True]},
    )
    def test__scaled_is_finite__proper_result(self, val, expected_out):
        out = scaled_is_finite(val)
        assert isinstance(out, Array)
        assert out.dtype == np.bool_
        npt.assert_array_equal(out, expected_out)

    @parameterized.parameters(
        {"bool_prim": lax.eq_p},
        {"bool_prim": lax.ne_p},
        {"bool_prim": lax.lt_p},
        {"bool_prim": lax.le_p},
        {"bool_prim": lax.gt_p},
        {"bool_prim": lax.ge_p},
    )
    def test__scaled_boolean_binary_op__proper_result(self, bool_prim):
        lhs = scaled_array(self.rs.rand(5), 2.0, dtype=np.float32)
        rhs = scaled_array(self.rs.rand(5), 3.0, dtype=np.float32)
        scaled_bool_op, _ = find_registered_scaled_op(bool_prim)
        out0 = scaled_bool_op(lhs, rhs)
        out1 = scaled_bool_op(lhs, lhs)
        assert isinstance(out0, Array)
        assert out0.dtype == np.bool_
        npt.assert_array_equal(out0, bool_prim.bind(lhs.to_array(), rhs.to_array()))
        npt.assert_array_equal(out1, bool_prim.bind(lhs.to_array(), lhs.to_array()))

    def test__scaled_select_n__proper_result(self):
        mask = self.rs.rand(5) > 0.5
        lhs = scaled_array(self.rs.rand(5), 2.0, dtype=np.float32)
        rhs = scaled_array(self.rs.rand(5), 4.0, dtype=np.float32)
        out = scaled_select_n(mask, lhs, rhs)
        assert isinstance(out, ScaledArray)
        assert out.dtype == np.float32
        # Max scale used.
        npt.assert_almost_equal(out.scale, 4)
        npt.assert_array_equal(out, np.where(mask, rhs, lhs))

    @parameterized.parameters(
        {"scale": 0.25},
        {"scale": 8.0},
    )
    def test__scaled_select__relu_grad_example(self, scale):
        @autoscale
        def relu_grad(g):
            return lax.select(g > 0, g, lax.full_like(g, 0))

        # Gradient with some scale.
        gin = scaled_array([1.0, 0.5], np.float32(scale), dtype=np.float32)
        gout = relu_grad(gin)
        # Same scale should be propagated to gradient output.
        assert isinstance(gout, ScaledArray)
        npt.assert_array_equal(gout.scale, gin.scale)
