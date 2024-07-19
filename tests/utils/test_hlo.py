# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import chex
import jax
import jax.numpy as jnp

import jax_scalify as jsa


class ScalifyHloUtilsTests(chex.TestCase):
    def test__hlo_util__parse_hlo_module(self):
        def matmul_fn(a, b):
            return jax.lax.dot(a, b.T)

        a = jax.core.ShapedArray((16, 48), jnp.float16)
        b = jax.core.ShapedArray((32, 48), jnp.float16)

        matmul_lowered = jax.jit(matmul_fn).lower(a, b)
        matmul_compiled = matmul_lowered.compile()

        ops = jsa.utils.parse_hlo_module(matmul_compiled)
        assert len(ops) >= 6
        # TODO: other tests???
        # jsa.utils.print_hlo_module(matmul_compiled, backend_cfg=True, indent=2)
