# Copyright (c) 2023 Graphcore Ltd. All rights reserved.

from jax import lax

from .scaled_ops import scaled_mul

scaled_ops_registry = {lax.mul_p: scaled_mul}
