# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import chex
import jax
import jax.numpy as jnp

from jax_scaled_arithmetics.core import ScaledArray, autoscale


class AutoScaleInterpreterTests(chex.TestCase):
    def test__identity(self):
        def func(x):
            return x

        asfunc = autoscale(func)

        scale = jnp.array(1.0)
        inputs = jnp.array([1.0, 2.0])
        expected = jnp.array([1.0, 2.0])

        scaled_inputs = ScaledArray(inputs, scale)
        scaled_outputs = asfunc(scaled_inputs)[0]

        assert jnp.allclose(scaled_outputs.to_array(), expected)

        jaxpr = jax.make_jaxpr(asfunc)(scaled_inputs).jaxpr

        assert jaxpr.invars[0].aval.shape == inputs.shape
        assert jaxpr.invars[1].aval.shape == ()

        assert jaxpr.outvars[0].aval.shape == expected.shape
        assert jaxpr.outvars[1].aval.shape == ()

    def test__mul(self):
        def func(x, y):
            return x * y

        asfunc = autoscale(func)

        x_in = jnp.array([-2.0, 2.0])
        x_scale = jnp.array(0.5)
        x = ScaledArray(x_in, x_scale)

        y_in = jnp.array([1.5, 1.5])
        y_scale = jnp.array(2.0)
        y = ScaledArray(y_in, y_scale)

        expected = jnp.array([-3.0, 3.0])

        out = asfunc(x, y)[0]

        assert jnp.allclose(out.to_array(), expected)
