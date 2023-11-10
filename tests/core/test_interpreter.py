# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt

from jax_scaled_arithmetics.core import ScaledArray, autoscale, register_scaled_op, scaled_array


class AutoScaleInterpreterTests(chex.TestCase):
    def test__register_scaled_op__error_if_already_registered(self):
        with self.assertRaises(KeyError):
            register_scaled_op(jax.lax.mul_p, lambda a, _: a)

    @chex.variants(with_jit=True, without_jit=True)
    def test__scaled_identity_function(self):
        def func(x):
            return x

        # Autoscale + (optional) jitting.
        asfunc = self.variant(autoscale(func))

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

    @chex.variants(with_jit=True, without_jit=True)
    def test__scaled_mul__no_attributes(self):
        def func(x, y):
            return x * y

        # Autoscale + (optional) jitting.
        asfunc = self.variant(autoscale(func))

        x = scaled_array([-2.0, 2.0], 0.5, dtype=np.float32)
        y = scaled_array([1.5, 1.5], 2, dtype=np.float32)
        expected = jnp.array([-3.0, 3.0])

        out = asfunc(x, y)
        assert isinstance(out, ScaledArray)
        npt.assert_array_almost_equal(out, expected)

    @chex.variants(with_jit=True, without_jit=True)
    def test__scaled_convert_element_type__attributes_passing(self):
        def func(x):
            return jax.lax.convert_element_type(x, np.float16)

        # Autoscale + (optional) jitting.
        asfunc = self.variant(autoscale(func))
        x = scaled_array([-4.0, 2.0], 0.5, dtype=np.float32)
        out = asfunc(x)
        assert isinstance(out, ScaledArray)
        assert out.dtype == np.float16
        npt.assert_array_almost_equal(out, x)
