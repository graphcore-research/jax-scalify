# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from functools import partial

import chex
import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
import numpy.testing as npt
from absl.testing import parameterized
from numpy.typing import NDArray

from jax_scalify.core import scaled_array, scalify
from jax_scalify.ops import cast_on_backward, cast_on_forward, reduce_precision_on_forward


class ReducePrecisionDtypeTests(chex.TestCase):
    @parameterized.parameters(
        {"ml_dtype": ml_dtypes.float8_e4m3fn},
        {"ml_dtype": ml_dtypes.float8_e5m2},
    )
    def test__reduce_precision_on_forward__consistent_rounding_down(self, ml_dtype):
        # Values potentially "problematic" in FP8.
        values: NDArray[np.float16] = np.array([17, -17, 8, 1, 9, 11, 18], np.float16)
        out = reduce_precision_on_forward(values, dtype=ml_dtype)
        expected_out = values.astype(ml_dtype)
        assert out.dtype == values.dtype
        npt.assert_array_equal(out, expected_out)

    @parameterized.parameters(
        {"ml_dtype": ml_dtypes.float8_e4m3fn},
        {"ml_dtype": ml_dtypes.float8_e5m2},
    )
    def test__reduce_precision_on_forward__scalify_compatiblity(self, ml_dtype):
        values: NDArray[np.float16] = np.array([17, -17, 8, 1, 9, 11, 18], np.float16)
        arr = scaled_array(values, np.float32(1))
        out = scalify(partial(reduce_precision_on_forward, dtype=ml_dtype))(arr)

        npt.assert_array_equal(out.scale, arr.scale)
        npt.assert_array_equal(out, np.asarray(arr.data).astype(ml_dtype))


class CastOnForwardBackwardTests(chex.TestCase):
    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.parameters(
        {"dtype": jnp.float16},
        # TODO: uncomment when JAX 0.4+ used
        # {"dtype": jnp.float8_e4m3fn},
        # {"dtype": jnp.float8_e5m2},
    )
    def test__cast_on_forward_backward__proper_results(self, dtype):
        # Values potentially "problematic" in FP8.
        values: NDArray[np.float16] = np.array([17, -17, 8, 1, 9, 11, 18], np.float16)
        out_on_fwd = self.variant(partial(cast_on_forward, dtype=dtype))(values)
        out_on_bwd = self.variant(partial(cast_on_backward, dtype=dtype))(values)

        assert out_on_fwd.dtype == dtype
        assert out_on_bwd.dtype == values.dtype
        npt.assert_array_equal(out_on_fwd, jax.lax.convert_element_type(values, dtype))
        npt.assert_array_equal(out_on_bwd, values)

    @parameterized.parameters(
        {"dtype": jnp.float16},
        # TODO: uncomment when JAX 0.4+ used
        # {"dtype": jnp.float8_e4m3fn},
        # {"dtype": jnp.float8_e5m2},
    )
    def test__cast_on_backward__grad__proper_results(self, dtype):
        def fn(val, with_cast):
            if with_cast:
                val = cast_on_backward(val, dtype=dtype)
            val = val * val
            return jax.lax.reduce_sum_p.bind(val, axes=(0,))

        # Values potentially "problematic" in FP8.
        values: NDArray[np.float32] = np.array([17, -17, 8, 1, 9, 11, 18], np.float32)
        # Backward pass => gradient.
        grads = jax.grad(partial(fn, with_cast=True))(values)
        grads_ref = jax.grad(partial(fn, with_cast=False))(values)

        assert grads.dtype == dtype
        assert grads_ref.dtype == values.dtype
        npt.assert_array_equal(grads, jax.lax.convert_element_type(grads_ref, dtype))
