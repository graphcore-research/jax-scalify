# JAX Scalify: end-to-end scaled arithmetic

[![tests](https://github.com/graphcore-research/jax-scalify/actions/workflows/tests.yaml/badge.svg)](https://github.com/graphcore-research/jax-scalify/actions/workflows/tests-public.yaml)
![PyPI version](https://img.shields.io/pypi/v/jax-scalify)
[![license](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/graphcore-research/jax-scalify/blob/main/LICENSE)
[![GitHub Repo stars](https://img.shields.io/github/stars/graphcore-research/jax-scalify)](https://github.com/graphcore-research/jax-scalify/stargazers)
<!-- [![codecov](https://codecov.io/gh/jax-scalify/branch/main/graph/badge.svg?token=bHOkKY5Fze)](https://codecov.io/gh/jax-scalify) -->

[**Installation**](#installation)
| [**Quickstart**](#quickstart)
| [**Documentation**](#documentation)

**📣 Scalify** has been accepted to [**ICML 2024 workshop WANT**](https://openreview.net/forum?id=4IWCHWlb6K)! 📣

**JAX Scalify** is a library implementing end-to-end scale propagation and scaled arithmetic, allowing easy training and inference of
deep neural networks in low precision (BF16, FP16, FP8).

Loss scaling, tensor scaling and block scaling have been widely used in the deep learning literature to unlock training and inference at lower precision. Most of these works focus on ad-hoc approaches around scaling of matrix multiplications (and sometimes reduction operations). `Scalify` is adopting a more systematic approach with end-to-end scale propagation, i.e. transforming the full computational graph into a `ScaledArray` graph where every operation has `ScaledArray` inputs and returns `ScaledArray`:

```python
@dataclass
class ScaledArray:
    # Main data component, in low precision.
    data: Array
    # Scale, usually scalar, in FP32 or E8M0.
    scale: Array

    def __array__(self) -> Array:
        # Tensor represented as a `ScaledArray`.
        return data * scale.astype(self.data.dtype)
```

The main benefits of the `scalify` approach are:

* Agnostic to neural-net model definition;
* Decoupling scaling from low-precision, reducing the computational overhead of dynamic rescaling;
* FP8 matrix multiplications and reductions as simple as a cast;
* Out-of-the-box support of FP16 (scaled) master weights and optimizer state;
* Composable with JAX ecosystem: [Flax](https://github.com/google/flax), [Optax](https://github.com/google-deepmind/optax), ...

## Installation

JAX Scalify can be directly installed from PyPi:
```bash
pip install jax-scalify
```
Please follow [JAX documentation](https://github.com/google/jax/blob/main/README.md#installation) for a proper JAX installation on GPU/TPU.

The latest version of JAX Scalify is available directly from Github:
```bash
pip install git+https://github.com/graphcore-research/jax-scalify.git
```

## Quickstart

A typical JAX training loop just requires a couple of modifications to take advantage of `scalify`. More specifically:

* Represent input and state as `ScaledArray` using the `as_scaled_array` method (or variations of it);
* End-to-end scale propagation in `update` training method using `scalify` decorator;
* (Optionally) add `dynamic_rescale` calls to improve low-precision accuracy and stability;


The following (simplified) example presents how to `scalify` can be incorporated into a JAX training loop.
```python
import jax_scalify as jsa

# Scalify transform on FWD + BWD + optimizer.
# Propagating scale in the computational graph.
@jsa.scalify
def update(state, data, labels):
    # Forward and backward pass on the NN model.
    loss, grads =
        jax.grad(model)(state, data, labels)
    # Optimizer applied on scaled state.
    state = optimizer.apply(state, grads)
    return loss, state

# Model + optimizer state.
state = (model.init(...), optimizer.init(...))
# Transform state to scaled array(s)
sc_state = jsa.as_scaled_array(state)

for (data, labels) in dataset:
    # If necessary (e.g. images), scale input data.
    data = jsa.as_scaled_array(data)
    # State update, with full scale propagation.
    sc_state = update(sc_state, data, labels)
    # Optional dynamic rescaling of state.
    sc_state = jsa.ops.dynamic_rescale_l2(sc_state)
```
As presented in the code above, the model state is represented as a JAX PyTree of `ScaledArray`, propagated end-to-end through the model (forward and backward passes) as well as the optimizer.


A full collection of examples is available:
* [Scalify quickstart notebook](./examples/scalify-quickstart.ipynb): basics of `ScaledArray` and `scalify` transform;
* [MNIST FP16 training example](./examples/mnist/mnist_classifier_from_scratch.py): adapting JAX MNIST example to `scalify`;
* [MNIST FP8 training example](./examples/mnist/mnist_classifier_from_scratch_fp8.py): easy FP8 support in `scalify`;
* [MNIST Flax example](./examples/mnist/mnist_classifier_mlp_flax.py): `scalify` Flax training, with Optax optimizer integration;

## Documentation

* [**Scalify ICML 2024 workshop WANT paper**](https://openreview.net/forum?id=4IWCHWlb6K)
* [Operators coverage in JAX `scalify`](docs/operators.md)

## Development

For a local development setup, we recommend an interactive install:
```bash
git clone git@github.com:graphcore-research/jax-scalify.git
pip install -e ./
```

Running `pre-commit` and `pytest` on the JAX Scalify repository:
```bash
pip install pre-commit
pre-commit run --all-files
pytest -v ./tests
```
Python wheel can be built with the usual command `python -m build`.
