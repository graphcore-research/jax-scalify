# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any, Tuple

# import chex
import jax
import jax.numpy as jnp
import numpy as np

# Type aliasing. To be compatible with JAX 0.3 as well.
try:
    from jax import Array

    ArrayTypes: Tuple[Any, ...] = (Array,)
except ImportError:
    from jaxlib.xla_extension import DeviceArray as Array

    # Older version of JAX <0.4
    ArrayTypes = (Array, jax.interpreters.partial_eval.DynamicJaxprTracer)

try:
    from jax.stages import ArgInfo

    # Additional ArgInfo in recent JAX versions.
    ArrayTypes = (*ArrayTypes, ArgInfo)
except ImportError:
    pass


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
