# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import chex
import ml_dtypes
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scalify.quantization.scale import as_e8m0, from_e8m0, pow2_truncate


class QuantizationScaleTests(chex.TestCase):
    @parameterized.parameters(
        {"dtype": np.float16},
        {"dtype": np.float32},
        {"dtype": ml_dtypes.bfloat16},
    )
    def test__pow2_truncate__proper_result(self, dtype):
        vin = np.array([-2, 0, 2, 1, 9, 15]).astype(dtype)
        vout = pow2_truncate(vin)
        assert vout.dtype == vin.dtype
        npt.assert_array_equal(vout, [-2.0, 0.0, 2.0, 1.0, 8.0, 8.0])

    @parameterized.parameters(
        # {"dtype": np.float16},
        {"dtype": np.float32},
        {"dtype": ml_dtypes.bfloat16},
    )
    def test__as_e8m0__positive_values(self, dtype):
        vin = np.array([0.6, 2, 1, 9, 15, 127]).astype(dtype).reshape((-1, 2))
        vout = as_e8m0(vin)
        assert vout.dtype == np.uint8
        assert vout.shape == vin.shape
        npt.assert_array_equal(vout, np.log2(pow2_truncate(vin)) + 127)

    @parameterized.parameters(
        # {"dtype": np.float16},
        {"dtype": np.float32},
        {"dtype": ml_dtypes.bfloat16},
    )
    def test__as_e8m0__negative_values(self, dtype):
        vin = np.array([-0.1, -3, 0, 2**-127]).astype(dtype)
        vout = as_e8m0(vin)
        assert vout.dtype == np.uint8
        # NOTE: uint8(0) is the smallest positive scale in E8M0.
        npt.assert_array_equal(vout, np.uint8(0))

    @parameterized.parameters(
        # {"dtype": np.float16},
        {"dtype": np.float32},
        {"dtype": ml_dtypes.bfloat16},
    )
    def test__from_e8m0(self, dtype):
        vin = np.array([2**-127, 0.25, 1, 2, 8, 2**127.0]).astype(dtype).reshape((-1, 2))
        vin_e8m0 = as_e8m0(vin)
        vout = from_e8m0(vin_e8m0, dtype)
        assert vin.dtype == vout.dtype
        assert vout.shape == vin.shape
        npt.assert_array_equal(vout, vin)
