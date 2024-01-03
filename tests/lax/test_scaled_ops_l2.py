# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax import lax

from jax_scaled_arithmetics.core import ScaledArray, find_registered_scaled_op, scaled_array
from jax_scaled_arithmetics.lax import scaled_div, scaled_dot_general, scaled_mul, scaled_reduce_window_sum


class ScaledTranslationDotPrimitivesTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    @parameterized.parameters(
        {"ldtype": np.float32, "rdtype": np.float32},
        # {"ldtype": np.float32, "rdtype": np.float16}, # Not supported in JAX 0.3.x
        # {"ldtype": np.float16, "rdtype": np.float32},
        {"ldtype": np.float16, "rdtype": np.float16},
    )
    def test__scaled_dot_general__proper_scaling(self, ldtype, rdtype):
        # Reduction dimension: 5 => sqrt(5) ~ 2
        lhs = scaled_array(self.rs.rand(3, 5), 2.0, dtype=ldtype)
        rhs = scaled_array(self.rs.rand(5, 2), 4.0, dtype=rdtype)

        dimension_numbers = (((1,), (0,)), ((), ()))
        out = scaled_dot_general(lhs, rhs, dimension_numbers)
        expected_out = lax.dot_general(np.asarray(lhs), np.asarray(rhs), dimension_numbers)

        assert isinstance(out, ScaledArray)
        assert out.dtype == expected_out.dtype
        assert out.scale.dtype == np.float32  # TODO: more test coverage.
        npt.assert_almost_equal(out.scale, lhs.scale * rhs.scale * 2)
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
        x = scaled_array([-1.0, 2.0], 4.0, dtype=np.float32)
        y = scaled_array([1.5, 4.5], 2.0, dtype=np.float32)
        z = scaled_op(x, y)
        assert isinstance(z, ScaledArray)
        assert z.dtype == x.dtype
        # Round down to power-of-2
        npt.assert_almost_equal(z.scale, 4)

    @parameterized.parameters(
        {"prim": lax.add_p},
        {"prim": lax.sub_p},
    )
    def test__scaled_addsub__not_overflowing_scale(self, prim):
        scaled_op, _ = find_registered_scaled_op(prim)
        x = scaled_array([-1.0, 2.0], np.float16(2.0), dtype=np.float16)
        y = scaled_array([1.5, 4.0], np.float16(1024.0), dtype=np.float16)
        z = scaled_op(x, y)
        print(z, x, y)
        assert z.scale.dtype == np.float16
        assert np.isfinite(z.scale)
        npt.assert_array_almost_equal(z, prim.bind(np.asarray(x, np.float32), np.asarray(y, np.float32)), decimal=6)

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
        {"reduce_prim": lax.reduce_sum_p, "expected_scale": 2 * 2},
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
