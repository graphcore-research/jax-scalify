# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scaled_arithmetics.core import pow2_round_down, pow2_round_up
from jax_scaled_arithmetics.core.utils import _exponent_bits_mask, get_mantissa


class Pow2RoundingUtilTests(chex.TestCase):
    @parameterized.parameters(
        {"dtype": np.float16},
        {"dtype": np.float32},
    )
    def test__exponent_bitmask__inf_value(self, dtype):
        val = _exponent_bits_mask[np.dtype(dtype)].view(dtype)
        expected_val = dtype(np.inf)
        npt.assert_equal(val, expected_val)

    @parameterized.product(
        val_mant=[(1, 1), (2.1, 1.05), (0, 0), (0.51, 1.02), (65504, 1.9990234375)],
        dtype=[np.float16, np.float32, np.float64],
    )
    def test__get_mantissa__proper_value__multi_dtypes(self, val_mant, dtype):
        val, mant = dtype(val_mant[0]), dtype(val_mant[1])
        val_mant = get_mantissa(val)
        assert val_mant.dtype == val.dtype
        assert val_mant.shape == ()
        assert type(val_mant) in {type(val), np.ndarray}
        npt.assert_equal(val_mant, mant)
        # Should be consistent with `pow2_round_down`. bitwise, not approximation.
        npt.assert_equal(mant * pow2_round_down(val), val)

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
