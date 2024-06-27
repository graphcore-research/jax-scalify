# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
import chex
import numpy as np

import jax_scalify as jsa


class ScalifyTreeUtilTests(chex.TestCase):
    def test__tree_flatten__proper_result(self):
        values = {"a": np.int32(2), "b": jsa.as_scaled_array(np.float32(1.5), 1.0)}
        outputs, _ = jsa.tree.flatten(values)
        assert len(outputs) == 2
        assert outputs[0] == 2
        assert isinstance(outputs[1], jsa.ScaledArray)
        assert np.asarray(outputs[1]) == 1.5

    def test__tree_leaves__proper_result(self):
        values = {"a": np.int32(2), "b": jsa.as_scaled_array(np.float32(1.5), 1.0)}
        outputs = jsa.tree.leaves(values)
        assert len(outputs) == 2
        assert outputs[0] == 2
        assert isinstance(outputs[1], jsa.ScaledArray)
        assert np.asarray(outputs[1]) == 1.5

    def test__tree_structure__proper_result(self):
        values = {"a": np.int32(2), "b": jsa.as_scaled_array(np.float32(1.5), 1.0)}
        pytree = jsa.tree.structure(values)
        assert pytree == jsa.tree.flatten(values)[1]

    def test__tree_unflatten__proper_result(self):
        values_in = {"a": np.int32(2), "b": jsa.as_scaled_array(np.float32(1.5), 1.0)}
        outputs, pytree = jsa.tree.flatten(values_in)
        values_out = jsa.tree.unflatten(pytree, outputs)
        assert values_out == values_in

    def test__tree_map__proper_result(self):
        values = {"a": np.int32(2), "b": jsa.as_scaled_array(np.float32(1.5), 1.0)}
        outputs = jsa.tree.map(lambda v: v.dtype, values)
        assert outputs == {"a": np.int32, "b": np.float32}

    def test__tree_astype__all_leaves_casting(self):
        values = {"a": np.int32(2), "b": jsa.as_scaled_array(np.float32(1.5), 1.0)}
        outputs = jsa.tree.astype(values, dtype=np.float16)
        dtypes = jsa.tree.map(lambda v: v.dtype, outputs)
        assert dtypes == {"a": np.float16, "b": np.float16}

    def test__tree_astype__only_float_casting(self):
        values = {"a": np.int32(2), "b": jsa.as_scaled_array(np.float32(1.5), 1.0)}
        outputs = jsa.tree.astype(values, dtype=np.float16, floating_only=True)
        dtypes = jsa.tree.map(lambda v: v.dtype, outputs)
        assert dtypes == {"a": np.int32, "b": np.float16}
