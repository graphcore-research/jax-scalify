# JAX Scaled Arithmetics

JAX Scaled Arithmetics is a thin library implementing numerically stable scaled arithmetics, allowing easy training and inference of
deep neural networks in low precision (BF16, FP16, FP8).

* [Draft JSA design document](docs/design.md);

## Installation

Local git repository install:
```bash
git clone git@github.com:graphcore-research/jax-scaled-arithmetics.git
pip install -e ./
```

Running `pre-commit` and `pytest`:
```bash
pip install pre-commit
pre-commit run --all-files
pytest -v ./tests
```
