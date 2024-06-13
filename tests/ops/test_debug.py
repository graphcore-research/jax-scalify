# Copyright (c) 2023 Graphcore Ltd. All rights reserved.
import numpy as np

from jax_scalify.core import scaled_array, scalify
from jax_scalify.ops import debug_print


def test__debug_print__scaled_arrays(capfd):
    fmt = "INPUTS: {} + {}"

    def debug_print_fn(x):
        debug_print(fmt, x, x)

    input_scaled = scaled_array([2, 3], 2.0, dtype=np.float32)
    scalify(debug_print_fn)(input_scaled)
    # Check captured stdout and stderr!
    captured = capfd.readouterr()
    assert len(captured.err) == 0
    assert captured.out.strip() == fmt.format(input_scaled, input_scaled)
