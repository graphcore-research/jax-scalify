# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt

from jax_scaled_arithmetics.core import ScaledArray, scaled_array
from jax_scaled_arithmetics.ops import dynamic_rescale_l1, dynamic_rescale_l2, dynamic_rescale_max


class DynamicRescaleOpsTests(chex.TestCase):
    def test__dynamic_rescale_max__proper_max_rescale_pow2_rounding(self):
        arr_in = scaled_array([2, -3], np.float16(4), dtype=np.float16)
        arr_out = dynamic_rescale_max(arr_in)

        assert isinstance(arr_out, ScaledArray)
        assert arr_out.dtype == arr_in.dtype
        npt.assert_array_equal(arr_out.scale, np.float16(8))
        npt.assert_array_equal(arr_out, arr_in)

    def test__dynamic_rescale_l1__proper_l1_rescale_pow2_rounding(self):
        # L1 norm = 2
        arr_in = scaled_array([1, -6], np.float16(4), dtype=np.float16)
        arr_out = dynamic_rescale_l1(arr_in)

        assert isinstance(arr_out, ScaledArray)
        assert arr_out.dtype == arr_in.dtype
        npt.assert_array_equal(arr_out.scale, np.float16(8))
        npt.assert_array_equal(arr_out, arr_in)

    def test__dynamic_rescale_l2__proper_max_rescale_pow2_rounding(self):
        # L2 norm = 8.945
        arr_in = scaled_array([4, -8], np.float16(4), dtype=np.float16)
        arr_out = dynamic_rescale_l2(arr_in)

        assert isinstance(arr_out, ScaledArray)
        assert arr_out.dtype == arr_in.dtype
        npt.assert_array_equal(arr_out.scale, np.float16(16))
        npt.assert_array_equal(arr_out, arr_in)
