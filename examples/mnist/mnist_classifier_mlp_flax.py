# Copyright (c) 2024 Graphcore Ltd. All rights reserved.
"""A basic MNIST MLP training example using Flax and Optax.

Similar to JAX MNIST from scratch, but using Flax and Optax libraries.

This example aim is to show how Scalify can integrate with common
NN libraries such as Flax and Optax.
"""
import time
from functools import partial

import datasets
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax import linen as nn  # type:ignore

import jax_scalify as jsa

# from jax.scipy.special import logsumexp


def logsumexp(a, axis=None, keepdims=False):
    from jax import lax

    dims = (axis,)
    amax = jnp.max(a, axis=dims, keepdims=keepdims)
    # FIXME: not proper scale propagation, introducing NaNs
    # amax = lax.stop_gradient(lax.select(jnp.isfinite(amax), amax, lax.full_like(amax, 0)))
    amax = lax.stop_gradient(amax)
    out = lax.sub(a, amax)
    out = lax.exp(out)
    out = lax.add(lax.log(jnp.sum(out, axis=dims, keepdims=keepdims)), amax)
    return out


class MLP(nn.Module):
    """A simple 3 layers MLP model."""

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(features=512, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=512, use_bias=True)(x)
        x = nn.relu(x)
        x = nn.Dense(features=10, use_bias=True)(x)
        logprobs = x - logsumexp(x, axis=1, keepdims=True)
        return logprobs


def loss(model, params, batch):
    inputs, targets = batch
    preds = model.apply(params, inputs)
    # targets = jsa.lax.rebalance(targets, np.float32(1 / 8))
    return -jnp.mean(jnp.sum(preds * targets, axis=1))


def accuracy(model, params, batch):
    inputs, targets = batch
    target_class = jnp.argmax(targets, axis=1)
    preds = model.apply(params, inputs)
    predicted_class = jnp.argmax(preds, axis=1)
    return jnp.mean(predicted_class == target_class)


def update(model, optimizer, model_state, opt_state, batch):
    grads = jax.grad(partial(loss, model))(model_state, batch)
    # Optimizer update (state & gradients).
    updates, opt_state = optimizer.update(grads, opt_state, model_state)
    model_state = optax.apply_updates(model_state, updates)
    return model_state, opt_state


if __name__ == "__main__":
    step_size = 0.001
    num_epochs = 10
    batch_size = 128
    key = jax.random.PRNGKey(42)
    use_scalify: bool = True

    training_dtype = np.dtype(np.float16)
    optimizer_dtype = np.dtype(np.float16)
    scale_dtype = np.float32

    train_images, train_labels, test_images, test_labels = datasets.mnist()
    num_train = train_images.shape[0]
    num_complete_batches, leftover = divmod(num_train, batch_size)
    num_batches = num_complete_batches + bool(leftover)
    mnist_img_size = train_images.shape[-1]

    def data_stream():
        rng = np.random.RandomState(0)
        while True:
            perm = rng.permutation(num_train)
            for i in range(num_batches):
                batch_idx = perm[i * batch_size : (i + 1) * batch_size]
                yield train_images[batch_idx], train_labels[batch_idx]

    # Build model & initialize model parameters.
    model = MLP()
    model_state = model.init(key, np.zeros((batch_size, mnist_img_size), dtype=training_dtype))
    # Optimizer & optimizer state.
    # opt = optax.sgd(learning_rate=step_size)
    opt = optax.adam(learning_rate=step_size, eps=2**-16)
    opt_state = opt.init(model_state)
    # Freeze model, optimizer (with step size).
    update_fn = partial(update, model, opt)

    if use_scalify:
        # Transform parameters to `ScaledArray`.
        model_state = jsa.as_scaled_array(model_state, scale=scale_dtype(1.0))
        opt_state = jsa.as_scaled_array(opt_state, scale=scale_dtype(0.0001))
        # Scalify the update function as well.
        update_fn = jsa.scalify(update_fn)
    # Convert the model state (weights) & optimizer state to proper dtype.
    model_state = jsa.tree.astype(model_state, training_dtype)
    opt_state = jsa.tree.astype(opt_state, optimizer_dtype, floating_only=True)

    print(f"Using Scalify: {use_scalify}")
    print(f"Training data format: {training_dtype.name}")
    print(f"Optimizer data format: {optimizer_dtype.name}")
    print("")

    update_fn = jax.jit(update_fn)

    batches = data_stream()
    for epoch in range(num_epochs):
        start_time = time.time()

        for _ in range(num_batches):
            batch = next(batches)
            # Scaled micro-batch + training dtype cast.
            batch = jsa.tree.astype(batch, training_dtype)
            if use_scalify:
                batch = jsa.as_scaled_array(batch, scale=scale_dtype(1))
            with jsa.ScalifyConfig(rounding_mode=jsa.Pow2RoundMode.DOWN, scale_dtype=scale_dtype):
                model_state, opt_state = update_fn(model_state, opt_state, batch)

        epoch_time = time.time() - start_time

        # Evaluation in normal/unscaled float32, for consistency.
        unscaled_params = jsa.asarray(model_state, dtype=np.float32)
        train_acc = accuracy(model, unscaled_params, (train_images, train_labels))
        test_acc = accuracy(model, unscaled_params, (test_images, test_labels))
        print(f"Epoch {epoch} in {epoch_time:0.2f} sec")
        print(f"Training set accuracy {train_acc:0.5f}")
        print(f"Test set accuracy {test_acc:0.5f}")
