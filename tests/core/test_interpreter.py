# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from jax_scaled_arithmetics.core import ScaledArray, autoscale, scaled_array


class AutoScaleInterpreterTests(chex.TestCase):
    def test__identity(self):
        def func(x):
            return x

        asfunc = autoscale(func)

        scaled_inputs = scaled_array([1.0, 2.0], 1, dtype=np.float32)
        scaled_outputs = asfunc(scaled_inputs)
        expected = jnp.array([1.0, 2.0])

        assert isinstance(scaled_outputs, ScaledArray)
        npt.assert_array_almost_equal(scaled_outputs, expected)
        jaxpr = jax.make_jaxpr(asfunc)(scaled_inputs).jaxpr

        # Vars need to be primitive data types (e.g., f32) -> 2 Vars per ScaledArray
        assert jaxpr.invars[0].aval.shape == scaled_inputs.shape
        assert jaxpr.invars[1].aval.shape == ()

        assert jaxpr.outvars[0].aval.shape == expected.shape
        assert jaxpr.outvars[1].aval.shape == ()

    def test__mul(self):
        def func(x, y):
            return x * y

        asfunc = autoscale(func)

        x = scaled_array([-2.0, 2.0], 0.5, dtype=np.float32)
        y = scaled_array([1.5, 1.5], 2, dtype=np.float32)
        expected = jnp.array([-3.0, 3.0])

        out = asfunc(x, y)
        assert isinstance(out, ScaledArray)
        npt.assert_array_almost_equal(out, expected)
