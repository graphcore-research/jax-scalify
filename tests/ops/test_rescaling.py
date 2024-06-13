# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scalify.core import ScaledArray, scaled_array
from jax_scalify.ops import dynamic_rescale_l1, dynamic_rescale_l2, dynamic_rescale_max


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

    @parameterized.parameters(
        {"dynamic_rescale_fn": dynamic_rescale_max},
        {"dynamic_rescale_fn": dynamic_rescale_l1},
        {"dynamic_rescale_fn": dynamic_rescale_l2},
    )
    def test__dynamic_rescale__epsilon_norm_value(self, dynamic_rescale_fn):
        arr_in = scaled_array([0, 0], np.float32(1), dtype=np.float16)
        arr_out = dynamic_rescale_fn(arr_in)
        # Rough bounds on the epsilon value.
        assert arr_out.scale > 0.0
        assert arr_out.scale < 0.001
