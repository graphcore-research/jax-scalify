# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def get_numpy_api(val: Any) -> Any:
    """Get the Numpy API corresponding to an array.

    JAX or classic Numpy supported.
    """
    if isinstance(val, jax.Array):
        return jnp
    elif isinstance(val, (np.ndarray, np.number)):
        return np
    raise NotImplementedError(f"Unsupported input type '{type(val)}'. No matching Numpy API.")
