# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import ml_dtypes
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from numpy.typing import NDArray

from jax_scaled_arithmetics.core import autoscale, scaled_array
from jax_scaled_arithmetics.ops import cast_ml_dtype


class CastMLDtypeTests(chex.TestCase):
    @parameterized.parameters(
        {"ml_dtype": ml_dtypes.float8_e4m3fn},
        {"ml_dtype": ml_dtypes.float8_e5m2},
    )
    def test__cast_ml_dtype__consistent_rounding_down(self, ml_dtype):
        # Values potentially "problematic" in FP8.
        values: NDArray[np.float16] = np.array([17, -17, 8, 1, 9, 11, 18], np.float16)
        out = cast_ml_dtype(values, dtype=ml_dtype)
        expected_out = values.astype(ml_dtype)
        assert out.dtype == values.dtype
        npt.assert_array_equal(out, expected_out)

    @parameterized.parameters(
        {"ml_dtype": ml_dtypes.float8_e4m3fn},
        {"ml_dtype": ml_dtypes.float8_e5m2},
    )
    def test__cast_ml_dtype__autoscale_compatiblity(self, ml_dtype):
        values: NDArray[np.float16] = np.array([17, -17, 8, 1, 9, 11, 18], np.float16)
        arr = scaled_array(values, np.float32(1))
        out = autoscale(partial(cast_ml_dtype, dtype=ml_dtype))(arr)

        npt.assert_array_equal(out.scale, arr.scale)
        npt.assert_array_equal(out, np.asarray(arr.data).astype(ml_dtype))
