# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scaled_arithmetics.core import Pow2RoundMode, pow2_decompose, pow2_round_down, pow2_round_up
from jax_scaled_arithmetics.core.pow2 import _exponent_bits_mask, get_mantissa


class Pow2DecomposePrimitveTests(chex.TestCase):
    @parameterized.parameters(
        {"dtype": np.float16},
        {"dtype": np.float32},
    )
    def test__exponent_bitmask__inf_value(self, dtype):
        val = _exponent_bits_mask[np.dtype(dtype)].view(dtype)
        expected_val = dtype(np.inf)
        npt.assert_equal(val, expected_val)

    @parameterized.product(
        val_exp=[
            (0, 0),
            (1, 1),
            (2.1, 2),
            (0.3, 0.25),
            (0.51, 0.5),
            (65500, 32768),
            # Test float16 sub-normals.
            (np.finfo(np.float16).smallest_normal, np.finfo(np.float16).smallest_normal),
            (np.finfo(np.float16).smallest_subnormal, np.finfo(np.float16).smallest_subnormal),
            (np.float16(3.123283386230469e-05), 3.0517578e-05),
        ],
        dtype=[np.float16, np.float32],
        scale_dtype=[np.float16, np.float32],
    )
    def test__pow2_decompose_round_down__numpy_implementation__proper_result(self, val_exp, dtype, scale_dtype):
        scale_dtype = np.float32
        vin, exp_scale = dtype(val_exp[0]), scale_dtype(val_exp[1])
        scale, vout = pow2_decompose(vin, scale_dtype, Pow2RoundMode.DOWN)

        assert isinstance(scale, (np.ndarray, np.number))
        assert isinstance(vout, (np.ndarray, np.number))
        assert scale.dtype == scale_dtype
        assert vout.dtype == vin.dtype
        # Always accurate when casting up to scale dtype.
        npt.assert_equal(scale * vout.astype(scale_dtype), vin.astype(scale_dtype))
        npt.assert_equal(scale, exp_scale)

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        val_exp=[
            (0, 0),
            (1, 1),
            (2.1, 2),
            (0.3, 0.25),
            (0.51, 0.5),
            (65500, 32768),
            # Test float16 sub-normals.
            (np.finfo(np.float16).smallest_normal, np.finfo(np.float16).smallest_normal),
            (np.finfo(np.float16).smallest_subnormal, np.finfo(np.float16).smallest_subnormal),
            (np.float16(3.123283386230469e-05), 3.0517578e-05),
            # Test float32 sub-normals: known bug!
            # (np.finfo(np.float32).smallest_normal, np.finfo(np.float32).smallest_normal),
            # (np.finfo(np.float32).smallest_subnormal, np.finfo(np.float32).smallest_subnormal),
        ],
        dtype=[np.float16, np.float32],
        scale_dtype=[np.float16, np.float32],
    )
    def test__pow2_decompose_round_down__jax_numpy__proper_result(self, val_exp, dtype, scale_dtype):
        vin, exp_scale = dtype(val_exp[0]), scale_dtype(val_exp[1])
        vin = jnp.array(vin)
        scale, vout = self.variant(lambda v: pow2_decompose(v, scale_dtype, Pow2RoundMode.DOWN))(vin)

        assert isinstance(scale, jnp.ndarray)
        assert isinstance(vout, jnp.ndarray)
        assert scale.dtype == scale_dtype
        assert vout.dtype == vin.dtype
        # Always accurate when casting up to scale dtype.
        npt.assert_equal(np.asarray(scale), exp_scale)
        npt.assert_equal(scale * np.array(vout, scale_dtype), np.asarray(vin, scale_dtype))

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.product(
        val_exp=[
            (+np.inf, np.inf, +np.inf),
            (-np.inf, np.inf, -np.inf),
            (np.nan, np.inf, np.nan),  # FIXME? scale == np.inf?
        ],
        dtype=[np.float16, np.float32],
        scale_dtype=[np.float16, np.float32],
    )
    def test__pow2_decompose_round_down__special_values(self, val_exp, dtype, scale_dtype):
        vin, exp_scale, exp_vout = dtype(val_exp[0]), scale_dtype(val_exp[1]), dtype(val_exp[2])
        scale, vout = self.variant(partial(pow2_decompose, scale_dtype=scale_dtype, mode=Pow2RoundMode.DOWN))(vin)
        npt.assert_equal(np.ravel(scale)[0], exp_scale)
        npt.assert_equal(np.ravel(vout)[0], exp_vout)

    @parameterized.product(
        val_exp=[(0, 0), (1, 1), (2.1, 2), (0.3, 0.25), (0.51, 0.5), (65500, 32768)],
        dtype=[np.float16, np.float32, np.float64],
    )
    def test__pow2_round_down__proper_rounding__multi_dtypes(self, val_exp, dtype):
        val, exp = dtype(val_exp[0]), dtype(val_exp[1])
        pow2_val = pow2_round_down(val)
        assert pow2_val.dtype == val.dtype
        assert pow2_val.shape == ()
        assert type(pow2_val) in {type(val), np.ndarray}
        npt.assert_equal(pow2_val, exp)

    @parameterized.product(
        val_exp=[(2.1, 4), (0.3, 0.5), (0.51, 1), (17000, 32768)],
        dtype=[np.float16],
    )
    def test__pow2_round_up__proper_rounding__multi_dtypes(self, val_exp, dtype):
        val, exp = dtype(val_exp[0]), dtype(val_exp[1])
        pow2_val = pow2_round_up(val)
        assert pow2_val.dtype == val.dtype
        assert type(pow2_val) in {type(val), np.ndarray}
        npt.assert_equal(pow2_val, exp)

    @parameterized.product(
        val_mant=[(1, 1), (2.1, 1.05), (0, 0), (0.51, 1.02), (65504, 1.9990234375)],
        dtype=[np.float16, np.float32],  # FIXME: float64 support in pure Numpy
    )
    def test__get_mantissa__proper_value__multi_dtypes(self, val_mant, dtype):
        val, mant = dtype(val_mant[0]), dtype(val_mant[1])
        val_mant = get_mantissa(val)
        assert val_mant.dtype == val.dtype
        assert val_mant.shape == ()
        assert type(val_mant) in {type(val), np.ndarray}
        print(mant, val_mant, dtype)
        npt.assert_equal(val_mant, mant)
        # Should be consistent with `pow2_round_down`. bitwise, not approximation.
        npt.assert_equal(mant * pow2_round_down(val), val)
