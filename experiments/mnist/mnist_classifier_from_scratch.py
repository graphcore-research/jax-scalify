# Copyright 2018 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A basic MNIST example using Numpy and JAX.

The primary aim here is simplicity and minimal dependencies.
"""
import time
from functools import partial

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
from jax import grad, jit
from jax.scipy.special import logsumexp

import jax_scaled_arithmetics as jsa


def print_mean_std(name, v):
    data, scale = jsa.lax.get_data_scale(v)
    # Always use np.float32, to avoid floating errors in descaling + stats.
    v = jsa.asarray(data, dtype=np.float32)
    m, s = np.mean(v), np.std(v)
    # print(data)
    print(f"{name}: MEAN({m:.4f}) / STD({s:.4f}) / SCALE({scale:.4f})")


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]


def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        jsa.ops.debug_callback(partial(print_mean_std, "W"), w)
        jsa.ops.debug_callback(partial(print_mean_std, "B"), b)
        (w,) = jsa.ops.debug_callback_grad(partial(print_mean_std, "WGrad"), w)

        # Matmul + relu
        outputs = jnp.dot(activations, w) + b
        activations = jnp.maximum(outputs, 0)
        jsa.ops.debug_callback(partial(print_mean_std, "Act"), activations)
        # activations = jsa.ops.dynamic_rescale_l2_grad(activations)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w)
    jsa.ops.debug_callback(partial(print_mean_std, "Logits0"), logits)
    logits = logits + final_b

    jsa.ops.debug_callback(partial(print_mean_std, "Logits1"), logits)
    (logits,) = jsa.ops.debug_callback_grad(partial(print_mean_std, "LogitsGrad"), logits)

    logits = jsa.ops.dynamic_rescale_l2_grad(logits)
    # logits = logits.astype(np.float32)
    (logits,) = jsa.ops.debug_callback_grad(partial(print_mean_std, "LogitsGrad"), logits)

    logits = logits - logsumexp(logits, axis=1, keepdims=True)
    jsa.ops.debug_callback(partial(print_mean_std, "Logits2"), logits)
    # (logits,) = jsa.ops.debug_callback_grad(partial(print_mean_std, "LogitsGrad"), logits)
    return logits


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    # jsa.ops.debug_callback(partial(print_mean_std, "Preds"), preds)
    loss = jnp.sum(preds * targets, axis=1)
    # loss = jsa.ops.dynamic_rescale_l2(loss)
    # jsa.ops.debug_callback(partial(print_mean_std, "LOSS1"), loss)
    loss = -jnp.mean(loss)
    # jsa.ops.debug_callback(partial(print_mean_std, "LOSS2"), loss)
    return loss
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


if __name__ == "__main__":
    layer_sizes = [784, 1024, 1024, 10]
    param_scale = 2.0
    step_size = 0.001
    num_epochs = 10
    batch_size = 128

    training_dtype = np.float16
    scale_dtype = np.float32

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)

    def data_stream():
        rng = npr.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    batches = data_stream()
    params = init_random_params(param_scale, layer_sizes)
    # Transform parameters to `ScaledArray` and proper dtype.
    params = jsa.as_scaled_array(params, scale=scale_dtype(param_scale))
    params = jax.tree_map(lambda v: v.astype(training_dtype), params, is_leaf=jsa.core.is_scaled_leaf)

    # @jit
    @jsa.autoscale
    def update(params, batch):
        grads = grad(loss)(params, batch)
        return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]
        return [
            (jsa.ops.dynamic_rescale_l1(w - step_size * dw), jsa.ops.dynamic_rescale_l1(b - step_size * db))
            for (w, b), (dw, db) in zip(params, grads)
        ]

    num_batches = 1
    num_epochs = 1
    for epoch in range(num_epochs):
        # print("EPOCH:", epoch)
        start_time = time.time()
        for _ in range(num_batches):
            # print("BATCH...")
            batch = next(batches)
            # Scaled micro-batch + training dtype cast.
            batch = jsa.as_scaled_array(batch, scale=scale_dtype(1))
            batch = jax.tree_map(lambda v: v.astype(training_dtype), batch, is_leaf=jsa.core.is_scaled_leaf)

            with jsa.AutoScaleConfig(rounding_mode=jsa.Pow2RoundMode.DOWN, scale_dtype=scale_dtype):
                params = update(params, batch)

        epoch_time = time.time() - start_time

        # Evaluation in float32, for consistency.
        # raw_params = jsa.asarray(params, dtype=np.float32)
        # train_acc = accuracy(raw_params, (train_images, train_labels))
        # test_acc = accuracy(raw_params, (test_images, test_labels))
        # print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        # print(f"Training set accuracy {train_acc:0.5f}")
        # print(f"Test set accuracy {test_acc:0.5f}")
