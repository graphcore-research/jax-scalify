# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized

from jax_scaled_arithmetics.core import ScaledArray, autoscale, is_scaled_leaf, register_scaled_op, scaled_array
from jax_scaled_arithmetics.core.interpreters import promote_scalar_to_scaled_array


class AutoScaleInterpreterTests(chex.TestCase):
    def test__register_scaled_op__error_if_already_registered(self):
        with self.assertRaises(KeyError):
            register_scaled_op(jax.lax.mul_p, lambda a, _: a)

    @chex.variants(with_jit=True, without_jit=True)
    def test__autoscale_interpreter__normal_jax_mode(self):
        def func(x):
            return x * 2

        func = self.variant(autoscale(func))
        data = np.array([1, 2], dtype=np.float32)
        out = func(data)
        # Proper behaviour!
        assert isinstance(out, jax.Array)
        npt.assert_array_equal(out, [2, 4])
        # Check jaxpr.
        jaxpr = jax.make_jaxpr(func)(data).jaxpr
        assert len(jaxpr.invars) == 1
        assert len(jaxpr.outvars) == 1
        assert len(jaxpr.eqns) == 1

    def test__autoscale_interpreter__without_jit__proper_jaxpr_signature(self):
        def func(x):
            return x * 2

        scaled_func = autoscale(func)
        scaled_input = scaled_array([1.0, 2.0], 3, dtype=np.float32)
        jaxpr = jax.make_jaxpr(scaled_func)(scaled_input).jaxpr
        # Need 3 equations: 2 mul + 1 cast.
        assert len(jaxpr.eqns) == 3
        # Vars need to be primitive data types (e.g., f32) -> 2 Vars per ScaledArray
        assert jaxpr.invars[0].aval.shape == scaled_input.shape
        assert jaxpr.invars[1].aval.shape == ()
        assert jaxpr.outvars[0].aval.shape == scaled_input.shape
        assert jaxpr.outvars[1].aval.shape == ()

    def test__autoscale_interpreter__with_jit__proper_jaxpr_signature(self):
        def myfunc(x):
            return x * 2

        scaled_func = autoscale(jax.jit(myfunc))
        scaled_input = scaled_array([1.0, 2.0], 3, dtype=np.float32)
        jaxpr = jax.make_jaxpr(scaled_func)(scaled_input).jaxpr
        # One main jit equation.
        assert len(jaxpr.eqns) == 1
        eqn = jaxpr.eqns[0]
        assert eqn.primitive.name == "pjit"
        assert eqn.params["name"] == "myfunc"
        # TODO: other parameters.
        # Vars need to be primitive data types (e.g., f32) -> 2 Vars per ScaledArray
        assert jaxpr.invars[0].aval.shape == scaled_input.shape
        assert jaxpr.invars[1].aval.shape == ()
        assert jaxpr.outvars[0].aval.shape == scaled_input.shape
        assert jaxpr.outvars[1].aval.shape == ()

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        # Identity function!
        {"fn": lambda x: x, "inputs": [scaled_array([1.0, 2.0], 3, dtype=np.float32)]},
        # Non-trivial input JAX pytree.
        {
            "fn": lambda vals: vals["x"] * vals["y"],
            "inputs": [
                {
                    "x": scaled_array([1.0, 2.0], 3, dtype=np.float32),
                    "y": scaled_array([1.5, -2.5], 2, dtype=np.float32),
                }
            ],
        },
        # Non-trivial output JAX pytree
        {"fn": lambda x: {"x": (x,)}, "inputs": [scaled_array([1.0, 2.0], 3, dtype=np.float32)]},
        # Multi-inputs operation.
        {
            "fn": lambda x, y: x * y,
            "inputs": [scaled_array([-2.0, 0.5], 0.5, dtype=np.float32), scaled_array([1.5, 1.5], 2, dtype=np.float32)],
        },
        # Proper forwarding of attributes.
        {
            "fn": lambda x: jax.lax.convert_element_type(x, np.float16),
            "inputs": [scaled_array([-4.0, 2.0], 0.5, dtype=np.float32)],
        },
        # Proper constant scalar handling.
        {
            "fn": lambda x: x * 2,
            "inputs": [scaled_array([[-2.0, 0.5]], 0.5, dtype=np.float32)],
        },
        # Internal `pjit` primitive call.
        {
            "fn": jax.jit(lambda x, y: x * y),
            "inputs": [scaled_array([-2.0, 0.5], 0.5, dtype=np.float32), scaled_array([1.5, 1.5], 2, dtype=np.float32)],
        },
        # TODO/FIXME: Proper constant Numpy array handling.
        # {
        #     "fn": lambda x: x * np.array([2.0, 3.0], dtype=np.float32),
        #     "inputs": [scaled_array([[-2.0], [0.5]], 0.5, dtype=np.float32)],
        # },
    )
    def test__autoscale_decorator__proper_graph_transformation_and_result(self, fn, inputs):
        # Autoscale function + (optional) jitting.
        scaled_fn = self.variant(autoscale(fn))
        scaled_output = scaled_fn(*inputs)
        # Normal JAX path, without scaled arrays.
        raw_inputs = jax.tree_map(np.asarray, inputs, is_leaf=is_scaled_leaf)
        expected_output = self.variant(fn)(*raw_inputs)

        # Do we re-construct properly the output type (i.e. handling Pytree properly)?
        if not isinstance(expected_output, (np.ndarray, jax.Array)):
            assert type(scaled_output) is type(expected_output)

        # Check each output in the flatten tree.
        scaled_outputs_flat, _ = jax.tree_util.tree_flatten(scaled_output, is_leaf=is_scaled_leaf)
        expected_outputs_flat, _ = jax.tree_util.tree_flatten(expected_output)
        for scaled_out, exp_out in zip(scaled_outputs_flat, expected_outputs_flat):
            assert isinstance(scaled_out, ScaledArray)
            assert scaled_out.scale.shape == ()
            assert scaled_out.dtype == exp_out.dtype
            npt.assert_array_almost_equal(scaled_out, exp_out, decimal=4)

    @parameterized.parameters(
        {"input": np.array(3)},
        {"input": jnp.array(3)},
        {"input": 3},
        {"input": 3.0},
    )
    def test__promote_scalar_to_scaled_array__proper_output(self, input):
        scaled_val = promote_scalar_to_scaled_array(input)
        assert isinstance(scaled_val, ScaledArray)
        assert scaled_val.data.dtype == scaled_val.scale.dtype
        npt.assert_array_equal(scaled_val.data, 1)
        npt.assert_array_equal(scaled_val.scale, input)
