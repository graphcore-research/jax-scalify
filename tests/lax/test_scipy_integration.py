# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from jax import lax

from jax_scalify.core import scaled_array, scalify


class ScaledScipyHighLevelMethodsTests(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    def test__lax_full_like__zero_scale(self):
        def fn(a):
            return lax.full_like(a, 0)

        a = scaled_array(np.random.rand(3, 5).astype(np.float32), np.float32(1))
        scalify(fn)(a)
        # FIMXE/TODO: what should be the expected result?

    @chex.variants(with_jit=False, without_jit=True)
    @parameterized.parameters(
        {"dtype": np.float32},
        {"dtype": np.float16},
    )
    def test__scipy_logsumexp__accurate_scaled_op(self, dtype):
        from jax.scipy.special import logsumexp

        input_scaled = scaled_array(self.rs.rand(10), 4.0, dtype=dtype)
        # JAX `logsumexp` Jaxpr is a non-trivial graph!
        out_scaled = self.variant(scalify(logsumexp))(input_scaled)
        out_expected = logsumexp(np.asarray(input_scaled))
        assert out_scaled.dtype == out_expected.dtype
        # Proper accuracy + keep the same scale.
        npt.assert_array_equal(out_scaled.scale, input_scaled.scale)
        npt.assert_array_almost_equal(out_scaled, out_expected, decimal=5)
