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
    scaled_div,
    scaled_dot_general,
    scaled_is_finite,
    scaled_mul,
    scaled_pad,
    scaled_reduce_precision,
    scaled_reduce_window_sum,
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
            debug_callback(callback, a, a * 3)
            return a

        x = scaled_array(self.rs.rand(5), 2, dtype=np.float16)
        fn = self.variant(autoscale(fn))
        fn(x)

        assert len(host_values) == 2
        for sv in host_values:
            assert isinstance(sv, ScaledArray)
            npt.assert_array_equal(sv.data, x.data)
        npt.assert_array_equal(host_values[0].scale, x.scale)
        npt.assert_array_equal(host_values[1].scale, x.scale * 3)

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

    @parameterized.parameters(
        {"ldtype": np.float32, "rdtype": np.float32},
        # {"ldtype": np.float32, "rdtype": np.float16}, # Not supported in JAX 0.3.x
        # {"ldtype": np.float16, "rdtype": np.float32},
        {"ldtype": np.float16, "rdtype": np.float16},
    )
    def test__scaled_dot_general__proper_scaling(self, ldtype, rdtype):
        lhs = scaled_array(self.rs.rand(3, 5), 2.0, dtype=ldtype)
        rhs = scaled_array(self.rs.rand(5, 2), 3.0, dtype=rdtype)

        dimension_numbers = (((1,), (0,)), ((), ()))
        out = scaled_dot_general(lhs, rhs, dimension_numbers)
        expected_out = lax.dot_general(np.asarray(lhs), np.asarray(rhs), dimension_numbers)

        assert isinstance(out, ScaledArray)
        assert out.dtype == expected_out.dtype
        assert out.scale.dtype == np.float32  # TODO: more test coverage.
        npt.assert_almost_equal(out.scale, lhs.scale * rhs.scale * np.sqrt(5))
        npt.assert_array_almost_equal(out, expected_out, decimal=2)


class ScaledTranslationUnaryOpsTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        {"prim": lax.exp_p, "dtype": np.float16, "expected_scale": 1.0},  # FIXME!
        {"prim": lax.log_p, "dtype": np.float16, "expected_scale": 1.0},  # FIXME!
        {"prim": lax.neg_p, "dtype": np.float16, "expected_scale": 2.0},
        {"prim": lax.abs_p, "dtype": np.float16, "expected_scale": 2.0},
        {"prim": lax.cos_p, "dtype": np.float16, "expected_scale": 1.0},
        {"prim": lax.sin_p, "dtype": np.float16, "expected_scale": 1.0},
    )
    def test__scaled_unary_op__proper_result_and_scaling(self, prim, dtype, expected_scale):
        scaled_op, _ = find_registered_scaled_op(prim)
        val = scaled_array(self.rs.rand(3, 5), 2.0, dtype=dtype)
        out = self.variant(scaled_op)(val)
        expected_output = prim.bind(np.asarray(val))
        assert isinstance(out, ScaledArray)
        assert out.dtype == val.dtype
        assert out.scale.dtype == val.scale.dtype
        npt.assert_almost_equal(out.scale, expected_scale)
        npt.assert_array_almost_equal(out, expected_output)


class ScaledTranslationBinaryOpsTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        prim=[lax.add_p, lax.sub_p, lax.mul_p, lax.div_p, lax.min_p, lax.max_p],
        dtype=[np.float16, np.float32],
        sdtype=[np.float16, np.float32],
    )
    def test__scaled_binary_op__proper_result_and_promotion(self, prim, dtype, sdtype):
        scaled_op, _ = find_registered_scaled_op(prim)
        # NOTE: direct construction to avoid weirdity between NumPy array and scalar!
        x = ScaledArray(np.array([-1.0, 2.0], dtype), sdtype(3.0))
        y = ScaledArray(np.array([1.5, 4.5], dtype), sdtype(2.0))
        # Ensure scale factor has the right dtype.
        assert x.scale.dtype == sdtype
        assert y.scale.dtype == sdtype

        z = self.variant(scaled_op)(x, y)
        expected_z = prim.bind(np.asarray(x), np.asarray(y))

        assert z.dtype == x.dtype
        assert z.scale.dtype == sdtype
        npt.assert_array_almost_equal(z, expected_z, decimal=3)

    @parameterized.parameters(
        {"prim": lax.add_p},
        {"prim": lax.sub_p},
    )
    def test__scaled_addsub__proper_scaling(self, prim):
        scaled_op, _ = find_registered_scaled_op(prim)
        x = scaled_array([-1.0, 2.0], 3.0, dtype=np.float32)
        y = scaled_array([1.5, 4.5], 2.0, dtype=np.float32)
        z = scaled_op(x, y)
        assert isinstance(z, ScaledArray)
        assert z.dtype == x.dtype
        npt.assert_almost_equal(z.scale, np.sqrt(4.0 + 9.0))

    def test__scaled_mul__proper_scaling(self):
        x = scaled_array([-2.0, 2.0], 3, dtype=np.float32)
        y = scaled_array([1.5, 1.5], 2, dtype=np.float32)
        z = scaled_mul(x, y)
        assert isinstance(z, ScaledArray)
        assert z.scale == 6
        npt.assert_array_almost_equal(z, np.asarray(x) * np.asarray(y))

    def test__scaled_div__proper_scaling(self):
        x = scaled_array([-2.0, 2.0], 3.0, dtype=np.float32)
        y = scaled_array([1.5, 1.5], 2.0, dtype=np.float32)
        z = scaled_div(x, y)
        assert isinstance(z, ScaledArray)
        assert z.scale == 1.5
        npt.assert_array_almost_equal(z, np.asarray(x) / np.asarray(y))


class ScaledTranslationReducePrimitivesTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    @parameterized.parameters(
        {"reduce_prim": lax.reduce_sum_p, "expected_scale": 2 * np.sqrt(5)},
        {"reduce_prim": lax.reduce_prod_p, "expected_scale": 2**5},
        {"reduce_prim": lax.reduce_min_p, "expected_scale": 2},
        {"reduce_prim": lax.reduce_max_p, "expected_scale": 2},
    )
    def test__scaled_reduce__single_axis__proper_scaling(self, reduce_prim, expected_scale):
        axes = (0,)
        # NOTE: float16 useful for checking dtype promotion!
        val = scaled_array(self.rs.rand(5), 2.0, dtype=np.float16)
        scaled_reduce_op, _ = find_registered_scaled_op(reduce_prim)
        out = scaled_reduce_op(val, axes=axes)

        assert isinstance(out, ScaledArray)
        assert out.shape == ()
        assert out.dtype == val.dtype
        npt.assert_almost_equal(out.scale, expected_scale)
        npt.assert_array_almost_equal(out, reduce_prim.bind(np.asarray(val), axes=axes))

    def test__scaled_reduce_window_sum__proper_result(self):
        val = scaled_array(self.rs.rand(5), 2.0, dtype=np.float32)
        out = scaled_reduce_window_sum(
            val,
            window_dimensions=(3,),
            window_strides=(1,),
            padding=((1, 0),),
            base_dilation=(1,),
            window_dilation=(1,),
        )
        assert isinstance(out, ScaledArray)
        assert out.shape == (4,)
        assert out.dtype == val.dtype
        npt.assert_almost_equal(out.scale, val.scale)


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
        rhs = scaled_array(self.rs.rand(5), 3.0, dtype=np.float32)
        out = scaled_select_n(mask, lhs, rhs)
        assert isinstance(out, ScaledArray)
        assert out.dtype == np.float32
        npt.assert_almost_equal(out.scale, 1)  # FIXME!
        npt.assert_array_equal(out, np.where(mask, rhs, lhs))
