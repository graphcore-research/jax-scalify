<!-- pandoc ./docs/stable-scaled-arithmetics.md --pdf-engine=xelatex -o ./docs/stable-scaled-arithmetics.pdf -V geometry:margin=2cm -->

# Scalify: stable scaled arithmetics

## Introduction

Low precision floating point arithmetics (`bloat16`, `float16`, `float8`, ...) are key to scaling up and speeding up large NN models. Nevertheless, instability in low precision is still an unsolved problem, as there is no clear systematic way of stabilizing NN training. Right now, the literature on the topic uses a combination of techniques:

* Automated loss scaling: scaling the loss to get gradient tensors in the `(b)float16` dynamic range;
* Bias selection for `float8` matmuls & convolutions (forward and backward passes);

Many large models articles are still arguing that `bfloat16` (or even worse `float16`) is just too unstable to get predictible training (see Gopher paper, ...).

Hence, there is stronger and stronger evidence additional refinement might be required to get there: loss scaling differentiated between activations & weights, re-scaling of partials in `sum` reductions, stable reduce mean, matmul unit scaling, ...

Ad-hoc fixes may work at short term for model training, but is not a sustainable solution in the long run for the ML community as frameworks integration, model tweaking, ... will become an increasingly complex implementation and maintenance burden.

**Is there a more systematic & consistent way of solving low precision arithmetics in ML?**

There is increasing evidence that `float16` with scale bias (`sfloat16`?) is a format with enough precision and dynamic range to represent ML tensors. Additionally `sfloat8` is also a very good option for (matmul/conv) activations, weights (and most probably gradients as well).

Hence, if `sfloat16` and `sfloat8` are proper representation types for ML tensors, can we define a stable set of arithmetic operators directly targeting scaled tensors? With the goal of having a general approach covering and extending:

* Automated loss scaling (ALS);
* Float8 automated bias selection (ABS);
* Unit scaling;

**What are the benefits?**

* Keep the same symbolic graph: compared to ALS or unit scaling, the mathematical graph is not changed;
* Systematic approach to floating point stability: not only applied to NN, but any computational graph;
* Explicit scale bias propagation, compared to unit scaling (no propagation) or ALS (implicit loss bias propagation);
* Unifying different techniques under the same abstraction;
* Reduce the ML framework integration complexity: every rescaling (static or dynamic) is purely local, no need anymore for global reduce of statistics like in ALS (which adds great complexity + communication cost, especially in pipelined models);
* Keep the model definition simple: no need to mingle together model graph and floating point stability techniques;
* Potential decoupling of storage type (static) & compute type (dynamically chosen);

### Comparison table vs existing methods

Summary of how it would compare to existing methods (credits to @thecharlieblake for this great table!):


|                                   | **Static Loss Scaling** | **Automatic Loss Scaling** | **Automatic Bias Selection** | **Unit Scaling** | **Auto-Scale** |
| --------------------------------- | ----------------------- | -------------------------- | ---------------------------- | ---------------- | -------------- |
| Scale propagates                  | (implicit)              | (implicit)                 |                              | (at init)        | x              |
| Automatic (over time)             |                         | x                          | x                            |                  | x              |
| Scale computed locally            |                         |                            | x                            | x                | x              |
| ~ perfect scale at init           |                         |                            | (possibly)                   | x                | x              |
| Doesn't alter  model              | x                       | x                          | x                            |                  | x              |
<!--
| Scale propagates                  | ✓                       | ✓                          |                              | (at init)        | ✓              |
| Automatic (changes over time)     |                         | ✓                          | ✓                            |                  | ✓              |
| Scale computed locally            |                         |                            | ✓                            | ✓                | ✓              |
| ~ perfect scale at initialisation |                         |                            | (possibly)                   | ✓                | ✓              |
| Doesn't alter underlying model    | ✓                       | ✓                          | ✓                            |                  | ✓              | -->


<!-- | Technique                         | Motivation | Implementation |
| --------------------------------- | -- | -- |
| Scale propagates                  | Changes in scale are typically passed on to subsequent operations in the fwd & bwd pass. If our numerics can also pass on shifts in bias, we avoid the overhead in scaling / tracking each tensor individually. | The output bias of an operation is a function of its inputs' biases. |
| Automatic (changes over time)     | We expect the scale of our tensors to change from their values at initialisation (due to e.g. the distribution of weight values shifting; initial gradient signal diminishing). By updating our biases over time, we can ensure our numerics are always well-scaled. | Collect histograms every $n$ steps and re-compute the ideal bias, according to some scoring functions. |
| Scale computed locally            | Different tensors in a network (or in the general case, computational graph) may have different scales, and require different biases. This is particularly true for grad_ws and grad_xs. | Each layer / sub-graph / tensor has its own independent bias. |
| ~ perfect scale at initialisation | We can use the "unit scaling rules" to achieve approximately unit variance for the first fwd & bwd pass. Without this, our tensors immediately require re-scaling. Starting in the ideal range may mean few re-scalings are required. | Apply appropriate scaling factors to the output of our operations (as determined in the unit scaling work). | -->


## JAX Scalify: end-to-end scaled arithmetics

We focus here on a JAX implementation, but it should be possible to adopt a similar approach in Pytorch (with tracing or dynamo?). Modern ML frameworks expose their IR at the Python level, allowing users to perform complex transforms on the computational graph without any modification to the C++ backend (XLA, ...).

To support scaled arithmetics in JAX, we first need a simple data structure which can represent any scaled tensor:
```python
@chex.dataclass
class ScaledArray:
    # Data, in IEEE format, with no scaling.
    data: jax.DeviceArray
    # Scaling scalar (float32).
    scale: ScalarArray

    @property
    def aval(self):
        return self.data.aval

    @property
    def value(self):
        return self.data * self.scale
```
In addition to a general `scale` tensor, a couple of special constant values could also be supported for `scale`:

* `ScaledArray(data, 1)`: equivalent to a standard array;
* `ScaledArray(data, 0)`: representing a zero array;
* `ScaledArray(data, None)`: equivalent to `ScaledArray(data, 1)`;
*
These special cases could be used for specific symbolic optimisations when tracing the JAX graph and generating XLA.

**Note:** one can imagine a generalization of `ScaledArray` where `scale` is any array which can broadcasted to `data` shape. That would allow scaling per batch item or/and scaling per channel for instance.

Then, assuming a JAX computational function (e.g. typically forward + backward passes):
```python
def value_and_gradients(data: jax.Array, weights: Struct[jax.Array]) -> jax.Array, Struct[jax.Array]:
    #  jax.value_and_grad...
    return loss, grads
```
we want an automatic JAX graph transform `@auto_scale` generating a function with the following signature:
```python
scaled_value_and_gradients(data: ScaledArray, weights: Struct[ScaledArray])
    -> ScaledArray, Struct[ScaledArray]:
```
where `Struct` can be any Pytree compatible nested structure of JAX arrays.

As indicated above, the `@auto_scale` JAX transform would not alter the underlying mathematical definition of the model.

### Scaled JAX LAX API

Such an automated transform `@auto_scale` would be made possible thanks to the 1-to-1 mapping of JAX primitives to scaled primitives. JAX LAX represents the low level API which needs to be translated. Let's present here the implementation of a couple of major primitives, and how the scaling gets propagated.


**Matmul scaled primitive**
```python
def scaled_matmul(A: ScaledArray, B: ScaledArray,
                  static_scaling_strategy = unit_scaling_strategy) -> ScaledArray:
    # By default unit scaling, but could be customized depending on application/hw
    static_scaling = static_scaling_strategy(A, B)
    # Fast in IPU HW with pow2 unit scaling.
    C = A.data @ B.data * (1. / static_scaling)
    sC = A.scale * B.scale * static_scaling
    return ScaledArray(C, sC)
```
This operation would directly map into MK3 FP8/FP16 matmul with bias scaling. And unit scaling-like strategies would be used to deduce the output scale bias.

**Cast**
```python
def scaled_cast(A: ScaledArray, dtype: Any) -> ScaledArray:
    # Pure forwarding of scale.
    return ScaledArray(lax.cast(A.data, dtype), A.scale)
```
This op is just calling the normal casting, without any rescaling. Dynamic rescaling (e.g. FP8 ABS) would happen outside (and potentially be fused by the compiler).

**Add / sub**

```python
def scaled_add(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    sC = max(A.scale, B.scale)
    # Mapping into scaled add IPU instruction (f16v4mix?)
    C = (A.scale / sC) * A.data + (B.scale / sC) * B.data
    return ScaledArray(C, sC)
```
This strategy should minimize the risk of overflow. But similarly to `matmul`, one could introduce a customizable static scaling if necessary.

**Mul / div**

```python
def scaled_mul(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    # One of the operand is already a scalar?
    if A.is_scalar:
        return ScaledArray(B.data, B.scale * A.value)
    if B.is_scalar:
        return ScaledArray(A.data, A.scale * B.value)

    # Is it the optimal scaling?
    sC = A.scale * B.scale
    C = A.data * B.data
    return ScaledArray(C, sC)
```

**Max / Min**

```python
def scaled_maximum(A: ScaledArray, B: ScaledArray) -> ScaledArray:
    # Optimization in case of 0 scale: no need to estimate output scale.
    if A.scale == 0:
        return ScaledArray(lax.max(B.data, 0), B.scale)
    if B.scale == 0:
        return ScaledArray(lax.max(A.data, 0), A.scale)

    # Max scale, to avoid overflowing.
    sC = max(A.scale, B.scale)
    C = lax.max((A.scale / sC) * A.data, (B.scale / sC) * B.data)
    return ScaledArray(C, sC)
```

```python
def scaled_max(A: ScaledArray, axis: int = None) -> ScaledArray:
    # No need to change the scale.
    return ScaledArray(lax.max(A.data, axis=axis), A.scale)
```

### A few higher level ops

Let's think about a few higher level ops (not part of low level JAX LAX), and how JAX auto-scale tracing could generate the optimal Jaxpr.

**Relu**

Optimal implementation of scaled `relu`:
```python
def scaled_relu(A: ScaledArray) -> ScaledArray:
    # Keep the same scale, as assumed optimal for the input tensor.
    return ScaledArray(jnp.relu(A.data), A.scale)
```

Interestingly, [JAX implementation](https://jax.readthedocs.io/en/latest/_modules/jax/_src/nn/functions.html#relu)
of `relu` is using `lax.maximum`, and as consequence, according to the implementation of `scaled_maximum`, it should
generate the exact same Jaxpr.

**Gelu**

TODO

**Softmax**

Optimal implementation of scaled `softmax`:
```python
def scaled_softmax(A: ScaledArray) -> ScaledArray:
    Amax = lax.max(A.data)
    Aexp = lax.exp((A.data - Amax) * A.scale)
    # Optimal scaling?
    Asoftmax = Aexp
    sAsoftmax = 1.0 / float32(lax.sum(Aexp))
    return ScaledArray(Asoftmax, sAsoftmax)
```
Note: see section on "the great decoupling" on using `A.scale` to decide which floating arithmetics to use (`float16` or `float32`).

TODO: compared to code generation by JAX tracing?

### Dynamic bias scaling

How would dynamic bias scaling (i.e. extending ALS/ABS) would be implemented in this framework? Until now, all scaled LAX primitives presented have static bias scaling: it is independent of the data distribution in tensors, only depending on the shape and op type.

In order to integrate dynamic bias scaling, which would improve stability and reduce overflow/underflow risks, one key additional primitive on `ScaledArray` is required:
```python
def rebalance_scale(A: ScaledArray, s: ScalarArray) -> ScaledArray:
    # Rebalance between FP8/FP16 dynamic range and the scale factor.
    return ScaledArray((1. / s) * A.data, s * A.scale)
```
From the high level model perspective, this is a no-op. But from the floating representation point of view, the later is allowing to rebalance the floating point data in order to "re-centre" the data distribution optimally in the floating dynamic range (FP8 or FP16).

How would this been used in practice? Here is a piece of code which would perform ABS/ALS type of dynamic rescaling (with optional casting):
```python
# Analyse statistics of ScaledArray A.
s = find_optimal_scaling(A)
A = rebalance_scale(A, s)
# Potentially followed by a cast
A8 = scaled_cast(A)
```
The scaling strategy can be chosen in this case to have an optimal FP8 cast (e.g. minimize quantization squared error), but this formalization allows any kind of dynamic rescaling of tensors.


## Performance and implementation details

How do we achieve optimal hardware usage and performance for scaled arithmetics on accelerated hardware (IPU, GPU, ...)?

The first step would be to actually implement a more specialized `ScaledArray`, limiting the scale to powers of two:
```python
@chex.dataclass
class Pow2ScaledArray:
    # Data, in IEEE format, with no scaling.
    data: jax.DeviceArray
    # Scaling bias (int32).
    scale_bias: ScalarArray

    @property
    def aval(self):
        return self.data.aval
```
It aligns with most hardware vendors designs where optimized operations (`matmul`, `conv`, ...) have a parameterized scaling bias (i.e. corresponding to a power of two scale). Additionally, elementwise `mul` is usually optimized on hardware for powers of two, as it corresponds to a simple add on the floating exponent.


**What about operations not optimized in hardware?**

Scaled arithmetics are complexifying the computational graph. What would be the performance overhead? We should differentiate two types of modifications of the graph:

* Scale estimate/update: these are scalar ops, used to estimate the output scale bias. As pure scalar ops, they will be negligible compared to the rest of the workload (and on the IPU, in the context `Pow2ScaledArray` `int32` scale bias, could even be performed in the supervisor thread);
* Pre- and post-scaling of tensors data: if no optimized directly in the hardware, these elementwise scaling ops could be fused with other elementwise ops (activation, normalization, ...) in the graph (before or after). Current [XLA (or other backends) compilers are already very efficient](https://llvm.org/devmtg/2019-04/slides/TechTalk-Joerg-Automated_GPU_Kernel_Fusion_with_XLA.pdf) at finding elementwise ops and fusing them into a single kernel/vertex (and keeping values in registers).


# Thinking further: the great decoupling

We usually couple together datatype of tensors and the arithmetic unit (i.e. assembly instructions). But there is no fundamental reason the later could not be dynamically chosen at runtime: when compiling a static computational graph (e.g. Poplar graph), we only decide on the datatype of tensors and the high level operations called, but nothing is preventing the former to dynamically choose between `float8`, `float16` or `float32` IPU instructions.

Until now, this kind of dynamic arithmetic has not been implemented as the decision factor can be complex. But `ScaledArray` could offer a simple decision just based on the scaling factor. Let's consider the example of `exp` (using pseudo-code):
```python
TODO: choose between FP16 and FP32 depending on the scale?
```

<!--
### Primitives

JAX is built on a foundational collection of primitives (roughly mapping XLA), exposed in [JAX LAX](https://jax.readthedocs.io/en/latest/jax.lax.html). Every computational graph built in JAX is represented as a `Jaxpr`, which is just a long list of `primitives` calls.

The first step of supporting scaled arithmetics in JAX is to provide a set of scaled primitives, with 1-to-1 mapping to original primitives. More specifically:
```python
out1, ..., outN = primitive(in1, ..., inM, attributes)
```
has an associated primitive:
```python
out1, outb1, ..., outN, outbN = scaled_primitive(in1, inb1, ..., inM, inbM, attributes)
``` -->
<!--
### Scaled primitives implementation?

By default, scaled primitives can be automatically generated, calling the original primitive:
```python
def scaled_primitive(in1, inb1, ..., inM, inbM, attributes):
    # Call original implementation with scaled inputs. Zero output bias.
    return zip(primitive(in1 * 2^inb1, ..., inM * 2^inbM), (0.0, ...)
```
This default implementation would probably be enough for 90% of primitives.

The goal would be to improve the implementation of these primitives such that they provide good floating point stability with a minimal overhead. -->
