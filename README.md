# JAX Scaled Arithmetics

**JAX Scaled Arithmetics** is a thin library implementing numerically stable scaled arithmetics, allowing easy training and inference of
deep neural networks in low precision (BF16, FP16, FP8).

Loss scaling, tensor scaling and block scaling have been widely used in the deep learning literature to unlock training and inference at lower precision. Usually, these works have focused on ad-hoc approaches around scaling of matmuls (and sometimes reduction operations). The JSA library is adopting a more systematic approach by transforming the full computational graph into a `ScaledArray` graph, i.e. every operation taking `ScaledArray` inputs and returning `ScaledArray`, where the latter is a simple datastructure:
```python
@dataclass
class ScaledArray:
    data: Array
    scale: Array

    def to_array(self) -> Array:
        return data * scale
```

A typical JAX training loop requires just a few modifications to take advantage of `autoscale`:
```python
import jax_scaled_arithmetics as jsa

params = jsa.as_scaled_array(params)

@jit
@jsa.autoscale
def update(params, batch):
    grads = grad(loss)(params, batch)
    return opt_update(params, grads)

for batch in batches:
    batch = jsa.as_scaled_array(batch)
    params = update(params, batch)
```
In other words: model parameters and micro-batch are converted to `ScaledArray` objects, and the decorator `jsa.autoscale` properly transforms the graph into a scaled arithmetics graph (see the [MNIST examples](./experiments/mnist/) for more details).

There are multiple benefits to this systematic approach:

* The model definition is unchanged (i.e. compared to unit scaling);
* The dynamic rescaling logic can be moved to optimizer update phase, simplifying the model definition and state;
* Clean implementation, as a JAX interpreter, similarly to `grad`, `vmap`, ...
* Generalize to different quantization paradigms: `int8` per channel, `MX` block scaling, per tensor scaling;
* FP16 training is more stable?
* FP8 support out of the box?


## Installation

JSA library can be easily installed in Python virtual environnment:
```bash
git clone git@github.com:graphcore-research/jax-scaled-arithmetics.git
pip install -e ./
```
The main dependencies are `numpy`, `jax` and `chex` libraries.

**Note:** it is compatible with [experimental JAX on IPU](https://github.com/graphcore-research/jax-experimental), which can be installed in a Graphcore Poplar Python environnment:
```bash
pip install jax==0.3.16+ipu jaxlib==0.3.15+ipu.sdk320 -f https://graphcore-research.github.io/jax-experimental/wheels.html
```

## Documentation

* [Draft Scaled Arithmetics design document](docs/design.md);
* [Scaled operators coverage](docs/operators.md)

## Development

Running `pre-commit` and `pytest`:
```bash
pip install pre-commit
pre-commit run --all-files
pytest -v ./tests
```
