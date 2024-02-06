# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax
import jax.numpy as jnp
import numpy as np

from jax_scaled_arithmetics.core import ScaledArray, autoscale, scaled_array


class ScaledJaxNumpyFunctions(chex.TestCase):
    def setUp(self):
        super().setUp()
        # Use random state for reproducibility!
        self.rs = np.random.RandomState(42)

    @chex.variants(with_jit=True, without_jit=False)
    def test__numpy_mean__proper_gradient_scale_propagation(self):
        def mean_fn(x):
            # Taking the square to "force" ScaledArray gradient.
            # Numpy mean constant rescaling creating trouble on backward pass!
            return jax.grad(lambda v: jnp.mean(v * v))(x)

        # size = 8 * 16
        input_scaled = scaled_array(self.rs.rand(8, 16).astype(np.float32), np.float32(1))
        output_grad_scaled = self.variant(autoscale(mean_fn))(input_scaled)

        assert isinstance(output_grad_scaled, ScaledArray)
        # Proper scale propagation on the backward pass (rough interval)
        assert np.std(output_grad_scaled.data) >= 0.25
        assert np.std(output_grad_scaled.data) <= 1.0
        # "small" scale.
        assert output_grad_scaled.scale <= 0.01
