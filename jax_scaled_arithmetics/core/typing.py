# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any

# import chex
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

# Type aliasing. To be compatible with JAX 0.3 as well.
if jax.__version_info__[1] > 3:
    Array = jax.Array
    ArrayTypes = (jax.Array, jax.stages.ArgInfo)
else:
    Array = jaxlib.xla_extension.DeviceArray
    ArrayTypes = (jaxlib.xla_extension.DeviceArray, jax.interpreters.partial_eval.DynamicJaxprTracer)


def get_numpy_api(val: Any) -> Any:
    """Get the Numpy API corresponding to an array.

    Using the NumPy API whenever possible when tracing a JAX graph
    allows for simple constant folding optimization.

    JAX or classic Numpy supported.
    """
    if isinstance(val, (np.ndarray, np.number)):
        return np
    if isinstance(val, ArrayTypes):
        return jnp
    raise NotImplementedError(f"Unsupported input type '{type(val)}'. No matching Numpy API.")
