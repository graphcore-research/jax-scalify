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
# Modified by Graphcore Ltd 2024.

"""A basic CIFAR10 example using Numpy and JAX.

CIFAR10 training using MLP network + raw SGD optimizer.
"""
import time

import dataset_cifar10
import jax
import jax.numpy as jnp
import numpy as np
import numpy.random as npr
from jax import grad, jit, lax

import jax_scalify as jsa


def logsumexp(a, axis=None, keepdims=False):
    dims = (axis,)
    amax = jnp.max(a, axis=dims, keepdims=keepdims)
    # FIXME: not proper scale propagation, introducing NaNs
    # amax = lax.stop_gradient(lax.select(jnp.isfinite(amax), amax, lax.full_like(amax, 0)))
    amax = lax.stop_gradient(amax)
    out = lax.sub(a, amax)
    out = lax.exp(out)
    out = lax.add(lax.log(jnp.sum(out, axis=dims, keepdims=keepdims)), amax)
    return out


def init_random_params(scale, layer_sizes, rng=npr.RandomState(0)):
    return [(scale * rng.randn(m, n), scale * rng.randn(n)) for m, n, in zip(layer_sizes[:-1], layer_sizes[1:])]


def print_mean_std(name, v):
    data, scale = jsa.lax.get_data_scale(v)
    # Always use np.float32, to avoid floating errors in descaling + stats.
    v = jsa.asarray(data, dtype=np.float32)
    m, s = np.mean(v), np.std(v)
    # print(data)
    print(f"{name}: MEAN({m:.4f}) / STD({s:.4f}) / SCALE({scale:.4f})")


def predict(params, inputs):
    activations = inputs
    for w, b in params[:-1]:
        # Matmul + relu
        outputs = jnp.dot(activations, w) + b
        activations = jnp.maximum(outputs, 0)

    final_w, final_b = params[-1]
    logits = jnp.dot(activations, final_w) + final_b

    # Dynamic rescaling of the gradient, as logits gradient not properly scaled.
    logits = jsa.ops.dynamic_rescale_l2_grad(logits)
    output = logits - logsumexp(logits, axis=1, keepdims=True)

    return output


def loss(params, batch):
    inputs, targets = batch
    preds = predict(params, inputs)
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(predict(params, inputs), axis=1)
    return jnp.mean(predicted_class == target_class)


if __name__ == "__main__":
    width = 2048
    lr = 1e-4
    use_scalify = True
    scalify = jsa.scalify if use_scalify else lambda f: f

    layer_sizes = [3072, width, width, 10]
    param_scale = 1.0

    step_size = lr
    num_epochs = 10
    batch_size = 128
    training_dtype = np.float16
    scale_dtype = np.float32

    train_images, train_labels, test_images, test_labels = dataset_cifar10.cifar()
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
    if use_scalify:
        params = jsa.as_scaled_array(params, scale=scale_dtype(param_scale))
    params = jax.tree_util.tree_map(lambda v: v.astype(training_dtype), params, is_leaf=jsa.core.is_scaled_leaf)

    @jit
    @scalify
    def update(params, batch):
        grads = grad(loss)(params, batch)
        return [(w - step_size * dw, b - step_size * db) for (w, b), (dw, db) in zip(params, grads)]

    for epoch in range(num_epochs):
        start_time = time.time()
        for _ in range(num_batches):
            batch = next(batches)
            # Scaled micro-batch + training dtype cast.
            if use_scalify:
                batch = jsa.as_scaled_array(batch, scale=scale_dtype(param_scale))
            batch = jax.tree_util.tree_map(lambda v: v.astype(training_dtype), batch, is_leaf=jsa.core.is_scaled_leaf)

            with jsa.ScalifyConfig(rounding_mode=jsa.Pow2RoundMode.DOWN, scale_dtype=scale_dtype):
                params = update(params, batch)

        epoch_time = time.time() - start_time

        # Evaluation in float32, for consistency.
        raw_params = jsa.asarray(params, dtype=np.float32)
        train_acc = accuracy(raw_params, (train_images, train_labels))
        test_acc = accuracy(raw_params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc:0.5f}")
        print(f"Test set accuracy {test_acc:0.5f}")
