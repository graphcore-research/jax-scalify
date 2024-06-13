# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scalify.core.utils import Array, python_scalar_as_numpy, safe_div, safe_reciprocal


class SafeDivOpTests(chex.TestCase):
    @parameterized.parameters(
        {"lhs": np.float16(0), "rhs": np.float16(0)},
        {"lhs": np.float32(0), "rhs": np.float32(0)},
        {"lhs": np.float16(2), "rhs": np.float16(0)},
        {"lhs": np.float32(4), "rhs": np.float32(0)},
    )
    def test__safe_div__zero_div__numpy_inputs(self, lhs, rhs):
        out = safe_div(lhs, rhs)
        assert isinstance(out, (np.number, np.ndarray))
        assert out.dtype == lhs.dtype
        npt.assert_equal(out, 0)

    @parameterized.parameters(
        {"lhs": np.float16(0), "rhs": jnp.float16(0)},
        {"lhs": jnp.float32(0), "rhs": np.float32(0)},
        {"lhs": jnp.float16(2), "rhs": np.float16(0)},
        {"lhs": np.float32(4), "rhs": jnp.float32(0)},
    )
    def test__safe_div__zero_div__jax_inputs(self, lhs, rhs):
        out = safe_div(lhs, rhs)
        assert isinstance(out, Array)
        assert out.dtype == lhs.dtype
        npt.assert_almost_equal(out, 0)

    @parameterized.parameters(
        {"val": np.float16(0)},
        {"val": jnp.float16(0)},
    )
    def test__safe_reciprocal__zero_div(self, val):
        out = safe_reciprocal(val)
        assert out.dtype == val.dtype
        npt.assert_almost_equal(out, 0)


def test__python_scalar_as_numpy__proper_convertion():
    npt.assert_equal(python_scalar_as_numpy(False), np.bool_(False))
    npt.assert_equal(python_scalar_as_numpy(4), np.int32(4))
    npt.assert_equal(python_scalar_as_numpy(3.2), np.float32(3.2))
