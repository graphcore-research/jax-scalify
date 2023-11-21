# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax import lax

from jax_scaled_arithmetics.core import ScaledArray, find_registered_scaled_op, scaled_array
from jax_scaled_arithmetics.lax import (
    scaled_add,
    scaled_broadcast_in_dim,
    scaled_concatenate,
    scaled_convert_element_type,
    scaled_dot_general,
    scaled_mul,
    scaled_slice,
    scaled_sub,
    scaled_transpose,
)


class ScaledTranslationPrimitivesTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    def test__scaled_broadcast_in_dim__proper_scaling(self):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
        z = scaled_broadcast_in_dim(x, shape=(5, 1), broadcast_dimensions=(0,))
        assert isinstance(z, ScaledArray)
        npt.assert_array_equal(z.scale, x.scale)
        npt.assert_array_almost_equal(z.data, x.data.reshape((5, 1)))

    def test__scaled_concatenate__proper_scaling(self):
        x = scaled_array(self.rs.rand(2, 3), 0.5, dtype=np.float32)
        y = scaled_array(self.rs.rand(5, 3), 2, dtype=np.float32)
        z = scaled_concatenate([x, y], dimension=0)
        assert isinstance(z, ScaledArray)
        npt.assert_array_equal(z.scale, y.scale)
        npt.assert_array_almost_equal(z, np.concatenate([x, y], axis=0))

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

    def test__scaled_slice__proper_scaling(self):
        x = scaled_array(self.rs.rand(5), 2, dtype=np.float32)
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

    def test__scaled_add__proper_scaling(self):
        x = scaled_array([-1.0, 2.0], 3.0, dtype=np.float32)
        y = scaled_array([1.5, 4.5], 2.0, dtype=np.float32)
        z = scaled_add(x, y)
        assert isinstance(z, ScaledArray)
        assert z.dtype == x.dtype
        npt.assert_almost_equal(z.scale, np.sqrt(4.0 + 9.0))
        npt.assert_array_almost_equal(z, np.asarray(x) + np.asarray(y))

    def test__scaled_sub__proper_scaling(self):
        x = scaled_array([-1.0, 2.0], 3.0, dtype=np.float32)
        y = scaled_array([1.5, 4.5], 2.0, dtype=np.float32)
        z = scaled_sub(x, y)
        assert isinstance(z, ScaledArray)
        assert z.dtype == x.dtype
        npt.assert_almost_equal(z.scale, np.sqrt(4.0 + 9.0))
        npt.assert_array_almost_equal(z, np.asarray(x) - np.asarray(y))

    def test__scaled_dot_general__proper_scaling(self):
        lhs = scaled_array(self.rs.rand(3, 5), 2.0, dtype=np.float32)
        rhs = scaled_array(self.rs.rand(5, 2), 3.0, dtype=np.float32)
        out = scaled_dot_general(lhs, rhs, (((1,), (0,)), ((), ())))
        assert isinstance(out, ScaledArray)
        assert out.dtype == lhs.dtype
        npt.assert_almost_equal(out.scale, lhs.scale * rhs.scale * np.sqrt(5))
        npt.assert_array_almost_equal(out, np.asarray(lhs) @ np.asarray(rhs))


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
        val = scaled_array(self.rs.rand(5), 2.0, dtype=np.float32)
        scaled_reduce_op, _ = find_registered_scaled_op(reduce_prim)
        out = scaled_reduce_op(val, axes=axes)

        assert isinstance(out, ScaledArray)
        assert out.shape == ()
        assert out.dtype == val.dtype
        npt.assert_almost_equal(out.scale, expected_scale)
        npt.assert_array_almost_equal(out, reduce_prim.bind(np.asarray(val), axes=axes))
