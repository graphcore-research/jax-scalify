# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import chex
import jax
import jax.lax
import jax.numpy as jnp
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from numpy.typing import NDArray

from jax_scaled_arithmetics.core import (
    Array,
    AutoScaleConfig,
    Pow2RoundMode,
    ScaledArray,
    asarray,
    autoscale,
    get_autoscale_config,
    is_scaled_leaf,
    register_scaled_op,
    scaled_array,
)
from jax_scaled_arithmetics.core.interpreters import ScalifyTracerArray


class ScalifyTracerArrayTests(chex.TestCase):
    @parameterized.parameters(
        {"arr": True},
        {"arr": 2},
        {"arr": 3.0},
    )
    def test__scalify_tracer_array__init__from_python_value(self, arr):
        tracer_arr = ScalifyTracerArray(arr)
        assert tracer_arr.array == arr
        assert not tracer_arr.is_scaled_array
        assert tracer_arr.is_broadcasted_scalar == (tracer_arr.size == 1)
        assert not tracer_arr.is_broadcasted_zero
        assert tracer_arr.to_array() is tracer_arr.array

    @parameterized.parameters(
        {"arr": np.float32(2)},
        {"arr": np.array([1, 2])},
        {"arr": jnp.array([3, 4])},
    )
    def test__scalify_tracer_array__init__from_normal_array(self, arr):
        tracer_arr = ScalifyTracerArray(arr)
        assert tracer_arr.array is arr
        assert not tracer_arr.is_scaled_array
        assert tracer_arr.is_broadcasted_scalar == (tracer_arr.size == 1)
        assert not tracer_arr.is_broadcasted_zero
        assert tracer_arr.to_array() is tracer_arr.array
        # Basic properties.
        assert tracer_arr.shape == arr.shape
        assert tracer_arr.size == arr.size

    @parameterized.parameters(
        {"arr": np.float32(2), "expected_is_zero": False},
        {"arr": np.float32(0), "expected_is_zero": True},
        {"arr": np.array([0, 0]), "expected_is_zero": True},
        {"arr": np.array([0.0, 0.0]), "expected_is_zero": True},
        {"arr": scaled_array([1, 2.0], 0.0, npapi=np), "expected_is_zero": True},
        {"arr": scaled_array([0, 0.0], 1.0, npapi=np), "expected_is_zero": True},
        {"arr": jnp.array([0, 0]), "expected_is_zero": False},
    )
    def test__scalify_tracer_array__init__zero_broadcasted_array(self, arr, expected_is_zero):
        tracer_arr = ScalifyTracerArray(arr)
        assert tracer_arr.is_broadcasted_zero == expected_is_zero
        # Scaled array conversion => scale should be zero.
        scaled_arr = tracer_arr.to_scaled_array()
        if tracer_arr.is_broadcasted_zero and isinstance(scaled_arr, ScaledArray):
            assert scaled_arr.scale == 0

    @parameterized.parameters({"arr": scaled_array([1, 2], 3.0)})
    def test__scalify_tracer_array__init__from_scaled_array(self, arr):
        tracer_arr = ScalifyTracerArray(arr)
        assert tracer_arr.array is arr
        assert tracer_arr.is_scaled_array
        assert tracer_arr.to_scaled_array() is tracer_arr.array
        assert not tracer_arr.is_broadcasted_zero

    def test__scalify_tracer_array__init__is_broadcasted_scalar_kwarg(self):
        arr = scaled_array([1, 2], 3.0)
        assert ScalifyTracerArray(arr, is_broadcasted_scalar=True).is_broadcasted_scalar
        assert not ScalifyTracerArray(arr, is_broadcasted_scalar=False).is_broadcasted_scalar

    def test__scalify_tracer_array__init__is_broadcasted_zero_kwarg(self):
        arr = scaled_array([0, 1], 3.0)
        # NOTE: explicitly passing the argument, not checking the data!
        assert ScalifyTracerArray(arr, is_broadcasted_zero=True).is_broadcasted_scalar
        assert ScalifyTracerArray(arr, is_broadcasted_zero=True).is_broadcasted_zero
        assert not ScalifyTracerArray(arr, is_broadcasted_zero=False).is_broadcasted_scalar
        assert not ScalifyTracerArray(arr, is_broadcasted_zero=False).is_broadcasted_zero

    def test__scalify_tracer_array__flatten__proper_pytree(self):
        arr = scaled_array([1, 2], 3.0)
        tracer_arr_in = ScalifyTracerArray(arr, is_broadcasted_scalar=True, is_broadcasted_zero=True)
        # Proper round trip!
        flat_arrays, pytree = jax.tree_util.tree_flatten(tracer_arr_in)
        tracer_arr_out = jax.tree_util.tree_unflatten(pytree, flat_arrays)

        assert isinstance(tracer_arr_out, ScalifyTracerArray)
        assert tracer_arr_out.is_broadcasted_scalar == tracer_arr_in.is_broadcasted_scalar
        assert tracer_arr_out.is_broadcasted_zero == tracer_arr_in.is_broadcasted_zero
        npt.assert_array_equal(np.asarray(tracer_arr_out.array), np.asarray(tracer_arr_in.array))

    @parameterized.parameters(
        {"input": 3.0},
        {"input": np.float32(3.0)},
        {"input": np.array(3.0)},
        {"input": jnp.array(3.0)},
    )
    def test__scalify_tracer_array__to_scaled_array__scalar_input(self, input):
        scaled_val = ScalifyTracerArray(input).to_scaled_array()
        assert isinstance(scaled_val, ScaledArray)
        assert scaled_val.data.dtype == scaled_val.scale.dtype
        # NOTE: scale is a power-of-two.
        npt.assert_almost_equal(np.asarray(scaled_val), input)

    @parameterized.parameters(
        {"input": np.array(3)},
        {"input": jnp.array(3)},
        {"input": np.int32(2)},
    )
    def test__scalify_tracer_array__to_scaled_array__not_promoted_input(self, input):
        out = ScalifyTracerArray(input).to_scaled_array()
        assert out is input

    def test__scalify_tracer_array__to_scaled_array__broadcasted_scalar_input(self):
        data: NDArray[np.float16] = np.array([5, 5], dtype=np.float16)
        scaled_out = ScalifyTracerArray(data, is_broadcasted_scalar=True).to_scaled_array(scale_dtype=np.float32)

        assert isinstance(scaled_out, ScaledArray)
        assert scaled_out.dtype == data.dtype
        assert scaled_out.scale.dtype == np.float32
        npt.assert_almost_equal(scaled_out.scale, 4)
        npt.assert_array_equal(np.asarray(scaled_out), data)


class AutoScaleInterpreterTests(chex.TestCase):
    def test__register_scaled_op__error_if_already_registered(self):
        with self.assertRaises(KeyError):
            register_scaled_op(jax.lax.mul_p, lambda a, _: a)

    @chex.variants(with_jit=True, without_jit=True)
    def test__autoscale_interpreter__normal_jax_mode(self):
        def func(x):
            return x * 2

        func = self.variant(autoscale(func))
        data: NDArray[np.float32] = np.array([1, 2], dtype=np.float32)
        out = func(data)
        # Proper behaviour!
        assert isinstance(out, Array)
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
        # Need 4 equations: 1 pow2_decompose + 2 mul + 1 cast.
        assert len(jaxpr.eqns) in {
            4,
        }
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
        # TODO: additional mul in `safe_check_dtypes` mode.
        assert len(jaxpr.eqns) in {1, 2}
        eqn = jaxpr.eqns[0]
        assert eqn.primitive.name in ("pjit", "xla_call")
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
        if not isinstance(expected_output, (np.ndarray, Array)):
            assert type(scaled_output) is type(expected_output)

        # Check each output in the flatten tree.
        scaled_outputs_flat, _ = jax.tree_util.tree_flatten(scaled_output, is_leaf=is_scaled_leaf)
        expected_outputs_flat, _ = jax.tree_util.tree_flatten(expected_output)
        for scaled_out, exp_out in zip(scaled_outputs_flat, expected_outputs_flat):
            assert isinstance(scaled_out, ScaledArray)
            assert scaled_out.scale.shape == ()
            assert scaled_out.dtype == exp_out.dtype
            npt.assert_array_almost_equal(scaled_out, exp_out, decimal=4)

    @chex.variants(with_jit=True, without_jit=True)
    def test__autoscale_decorator__promotion_broadcasted_scalar_array(self):
        def fn(sa, b):
            # Forcing broadcasting before the `lax.mul`
            b = jax.lax.broadcast_in_dim(b, sa.shape, ())
            return sa * b

        sa = scaled_array([0.5, 1.0], np.float32(4.0), dtype=np.float32)
        b = jnp.array(4.0, dtype=np.float16)

        scaled_fn = self.variant(autoscale(fn))
        sout = scaled_fn(sa, b)
        expected_out = fn(np.asarray(sa), b)

        assert isinstance(sout, ScaledArray)
        # Proper output scale, with `b` treated as scaled scalar.
        npt.assert_equal(np.asarray(sout.scale), np.float32(16))
        npt.assert_array_equal(np.asarray(sout), expected_out)

    @chex.variants(with_jit=True, without_jit=True)
    def test__autoscale_decorator__custom_jvp__proper_graph_transformation_and_result(self):
        # JAX official `jvp` example.
        @jax.custom_jvp
        def f(x, y):
            return jnp.sin(x) * y

        @f.defjvp
        def f_jvp(primals, tangents):
            x, y = primals
            x_dot, y_dot = tangents
            primal_out = f(x, y)
            tangent_out = jnp.cos(x) * x_dot * y + jnp.sin(x) * y_dot
            return primal_out, tangent_out

        def fn(x, y):
            return jax.jvp(f, (x, y), (x, y))

        # `autoscale` on `custom_jvp` method.
        scaled_inputs = (
            scaled_array([-2.0, 0.5], 0.5, dtype=np.float32),
            scaled_array([1.5, -4.5], 2, dtype=np.float32),
        )
        scaled_primals, scaled_tangents = self.variant(autoscale(fn))(*scaled_inputs)
        # JAX default/expected values
        inputs = tuple(map(asarray, scaled_inputs))
        primals, tangents = self.variant(fn)(*inputs)

        assert isinstance(scaled_primals, ScaledArray)
        assert isinstance(scaled_tangents, ScaledArray)
        npt.assert_array_almost_equal(scaled_primals, primals)
        npt.assert_array_almost_equal(scaled_tangents, tangents)

    @chex.variants(with_jit=True, without_jit=True)
    def test__autoscale_decorator__custom_vjp__proper_graph_transformation_and_result(self):
        # JAX official `vjp` example.
        @jax.custom_vjp
        def f(x, y):
            return jnp.sin(x) * y

        def f_fwd(x, y):
            return f(x, y), (jnp.cos(x), jnp.sin(x), y)

        def f_bwd(res, g):
            cos_x, sin_x, y = res
            return (cos_x * g * y, sin_x * g)

        f.defvjp(f_fwd, f_bwd)

        def fn(x, y):
            primals, f_vjp = jax.vjp(f, x, y)
            return primals, f_vjp(x * y)

        # `autoscale` on `custom_jvp` method.
        scaled_inputs = (
            scaled_array([-2.0, 0.5], 0.5, dtype=np.float32),
            scaled_array([1.5, -4.5], 2, dtype=np.float32),
        )
        scaled_primals, scaled_grads = self.variant(autoscale(fn))(*scaled_inputs)
        # JAX default/expected values
        inputs = tuple(map(asarray, scaled_inputs))
        primals, grads = self.variant(fn)(*inputs)

        assert isinstance(scaled_primals, ScaledArray)
        npt.assert_array_almost_equal(scaled_primals, primals)
        for g, sg in zip(grads, scaled_grads):
            assert isinstance(sg, ScaledArray)
            npt.assert_array_almost_equal(sg, g)

    def test__autoscale_config__default_values(self):
        cfg = get_autoscale_config()
        assert isinstance(cfg, AutoScaleConfig)
        assert cfg.rounding_mode == Pow2RoundMode.DOWN
        assert cfg.scale_dtype is None

    def test__autoscale_config__context_manager(self):
        with AutoScaleConfig(rounding_mode=Pow2RoundMode.NONE, scale_dtype=np.float32):
            cfg = get_autoscale_config()
            assert isinstance(cfg, AutoScaleConfig)
            assert cfg.rounding_mode == Pow2RoundMode.NONE
            assert cfg.scale_dtype == np.float32

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        {"scale_dtype": np.float16},
        {"scale_dtype": np.float32},
    )
    def test__autoscale_config__scale_dtype_used_in_interpreter_promotion(self, scale_dtype):
        def fn(x):
            # Sub-normal "learning rate" => can create issue when converting to FP16 scaled array.
            # return x * 3.123283386230469e-05
            # FIXME: issue when using the smallest FP16 sub-normal!
            return x * (np.finfo(np.float16).smallest_subnormal * 2)

        expected_output = fn(np.float16(1))

        with AutoScaleConfig(scale_dtype=scale_dtype):
            scaled_input = scaled_array(np.array(2.0, np.float16), scale=scale_dtype(0.5))
            scaled_output = self.variant(autoscale(fn))(scaled_input)
            assert scaled_output.scale.dtype == scale_dtype
            npt.assert_equal(np.asarray(scaled_output, dtype=np.float32), expected_output)
