# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt

from jax_scaled_arithmetics.core import ScaledArray, scaled_array
from jax_scaled_arithmetics.lax import (
    scaled_broadcast_in_dim,
    scaled_convert_element_type,
    scaled_mul,
    scaled_slice,
    scaled_transpose,
)


class ScaledTranslationPrimitivesTests(chex.TestCase):
    def test__scaled_broadcast_in_dim__proper_scaling(self):
        x = scaled_array(np.random.rand(5), 2, dtype=np.float32)
        z = scaled_broadcast_in_dim(x, shape=(5, 1), broadcast_dimensions=(0,))
        assert isinstance(z, ScaledArray)
        npt.assert_array_equal(z.scale, x.scale)
        npt.assert_array_almost_equal(z.data, x.data.reshape((5, 1)))

    def test__scaled_convert_element_type__proper_scaling(self):
        x = scaled_array(np.random.rand(5), 2, dtype=np.float32)
        z = scaled_convert_element_type(x, new_dtype=np.float16)
        assert isinstance(z, ScaledArray)
        npt.assert_array_equal(z.scale, x.scale)
        npt.assert_array_almost_equal(z.data, x.data.astype(z.dtype))

    def test__scaled_transpose__proper_scaling(self):
        x = scaled_array(np.random.rand(3, 5), 2, dtype=np.float32)
        z = scaled_transpose(x, (1, 0))
        assert isinstance(z, ScaledArray)
        assert z.scale == x.scale
        npt.assert_array_almost_equal(z.data, x.data.T)

    def test__scaled_slice__proper_scaling(self):
        x = scaled_array(np.random.rand(5), 2, dtype=np.float32)
        z = scaled_slice(x, (1,), (4,), (2,))
        assert isinstance(z, ScaledArray)
        assert z.scale == x.scale
        npt.assert_array_almost_equal(z.data, x.data[1:4:2])

    def test__scaled_mul__proper_scaling(self):
        x = scaled_array([-2.0, 2.0], 3, dtype=np.float32)
        y = scaled_array([1.5, 1.5], 2, dtype=np.float32)
        z = scaled_mul(x, y)
        assert isinstance(z, ScaledArray)
        assert z.scale == 6
        npt.assert_array_almost_equal(z, np.asarray(x) * np.asarray(y))
