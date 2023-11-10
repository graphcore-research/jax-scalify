# Scale propagation proposal

AutoScale relies on predicting the scale of tensors in computational graphs. Two key questions we have are:

 - What rules & approximations can we use to predict scale?
   - Therefore what is the design contract for a scaled op.
 - When should we requantise, in order to reset to an empirical scale?

In this proposal, we mainly explore the first question, with a bias for simplicity over fidelity.

## Design alternatives

### 1. Scale is a worst-case bound

Choose scale such that there is no situation where in-range inputs give out-of-range outputs.

Note that the following maths assumes that data is in the range `[-1, 1]`.

| Operation | Scaling rule |
| --- | --- |
| `R = quantise(a)` | `R.scale = max(abs(a))` |
| `R = add(A, B)` | `R.scale = A.scale + B.scale` |
| `R = sub(A, B)` | `R.scale = A.scale + B.scale` |
| `R = mul(A, B)` | `R.scale = A.scale * B.scale` |
| `R = dot(A, B)` | `R.scale = A.scale * B.scale * inner_dim` |
| `R = div(A, B)` | undefined |
| `R = pow(A, B)` | undefined unless all `b > 0`, then `R.scale = pow(A.scale, B.scale)` |

**Thoughts:**
 - Easy to define
 - No distributional assumptions (only that scale defines the max)
 - Should be relatively consistent (e.g. `A + A` behaves the same as `2 * A`)
 - Increases risk of underflow, if scale recalculation isn't frequent enough
   - May require too-frequent recalculation

**Example of scale-pessimism:**

ScaledTensors `A` and `B` each contain 1000 Gaussian values, std=1. The scale of each might be 3 (due to the 3-sigma rule). We run `R = A * B`, so `R.scale = 9` from the table above. An average case scaling rule would have set `R.scale = 3`.

### 2. Scale is an average-case bound

Choose scale that, under simple distributional assumptions, predicts the actual scale of the output.

Note that the following maths assumes that `1` is the midpoint of the range of tensor data (e.g. floating point formats).

| Operation | Scaling rule |
| --- | --- |
| `R = quantise(a)` | `R.scale = sqrt((a**2).mean())` |
| `R = add(A, B)` | `R.scale = sqrt(A.scale**2 + B.scale**2)` |
| `R = sub(A, B)` | `R.scale = sqrt(A.scale**2 + B.scale**2)` |
| `R = mul(A, B)` | `R.scale = A.scale * B.scale` |
| `R = dot(A, B)` | `R.scale = A.scale * B.scale * sqrt(inner_dim)` |
| `R = div(A, B)` | undefined |
| `R = pow(A, B)` | undefined |

**Thoughts:**
 - Somewhat easy to define "just assume that your inputs are IID-Gaussian (or IID-something-else)", if that helps.
 - There are some inconsistencies (`A + A` doesn't behave the same as `2 * A`), but this is true for finite-precision numerics anyway!
 - For undefined cases, more thought required!

### 3. Use worse-case scale, track average-case scale

Track both worst-case scale and average-case scale in `ScaledTensor`. The worst-case scale is the one used for quantisation, while the average-case scale is metadata. This scheme behaves like worst-case scaling, and uses the difference between the average-case scale and the worst-case scale to determine when to requantise.

For example:

```
R = dot(A, B)

dtype -- E4M3
A.shape, B.shape -- (4096, 4096), (4096, 4096)
A.scale -- (64, 2)     (worst, average)
B.scale -- (16, 1)     (worst, average)
```

If we ran-scaled, we would set an output scale of `(64*16*4096, 2*1*sqrt(4096)) = (4194304, 128)`. Since the ratio of worst to average scales is `32768`, we are worried about underflow in our E4M3 format (which has a ratio of max to min normal that is less than this). Therefore we requantise `A` (the worse offender) on the way in to the op. Perhaps the new `A.scale = (8, 4)`, so our output scale is a more reasonable `(524288, 256)` with a ratio of `2048`.

_While writing this example, I had to go quite extreme — this is due to the E4M3 format, which still has quite a wide range. Requantisation would automatically happen much more frequently in integer or low-E formats, for this reason._

### A. Notes on undefined cases

There are cases where scale tracking alone isn't enough to save you. Consider an implementation of LayerNorm:

```
Y = (X - X.mean()) / (X.var() + 1e-6).sqrt()
```

We know that `Y.scale = 1` is fine. But we'd need some sophisticated theorem-proving to get this from the bunch of primitives!

Tracing through the computation:

```
(X - X.mean()).scale                              ~= X.scale
(X.var() + 1e-6).scale                            ~= X.scale**2
(X.var() + 1e-6).sqrt().scale                     - (undefined, need to know >=0), then, yes ~= X.scale
((X - X.mean()) / (X.var() + 1e-6).sqrt()).scale  - (undefined, need to relate numer to denom)
```

Propagating extra information through ScaledTensor could help some cases, e.g. a minimum bound on the values would ease `sqrt()`, `pow()`. But others, like LayerNorm's `div()` seem harder.

Some options:
 - (More) theorem proving
 - User-side promises `Y = with_scale(1.0, (X - X.mean()) / (X.var() + 1e-6).sqrt())`
 - Extract subgraphs containing these operations, where we use regular (unscaled) tensors & lift to higher-precision

## Proposal #1 (most similar to unit scaling)

 - Scale is set to an estimate of the uncentered standard deviation
 - Ops assume inputs are IID Gaussian
 - For floating point types, `dequantise(X) = X.data * X.scale`, since `1` is the center of the log-range
 - For integer types, perhaps `dequantise(X) = X.data * X.scale * 4 / INT_MAX`, to provide 4-sigma headroom(?)
 - When an op cannot make a reasonable estimate of scale (e.g. `pow`, `div`, `sqrt`), perform the operation in higher precision ("global" setting), and requantise the output.
   - Remove unnecessary `dequantise(quantise(X))`, e.g. sqrt->div in the LayerNorm example above
   - Provide `with_scale()` to the user to make a promise about scale, overriding the above logic

This does not address when to reset to an emprical scale because are estimates are too weak. It also suggests we shouldn't worry too much about minor inconsistencies e.g. `(A + A).scale = sqrt(2)*A.scale` versus `(2*A).scale = 2*A.scale`.

## Proposal #2 (worst-case with average-case tracking for renormalisation)

 - Scaled tensors keep `scale` (used for quantistaion, based on worst case analysis) and `expected_scale` (based on average case)
 - Quantise sets `scale = abs(max(x))`, `expected_scale = sqrt(mean(x**2))`
   - (Being a bit sloppy here, the quantisation scale should use `dtype.max`)
 - Ops compute the new `scale` based on worst-case logic, and estimate `expected_scale` based on average-case logic
 - Every input to every scaled op includes a runtime-conditional requantisation based on the ratio between scale and expected scale, and the range of the element dtype
   - _Alternative: perhaps this doesn't have to be runtime-conditional, if we don't propagate `expected_scale`, and instead statically propagate the ratio, `scale/expected_scale` (e.g. this increases by `*sqrt(inner_dim)` for a dot product)_
 - When an op cannot make a reasonable estimate of scale (e.g. `pow`, `div`, `sqrt`), perform the operation in higher precision ("global" setting), and requantise the output
   - Remove unnecessary `dequantise(quantise(X))`, e.g. sqrt->div in the LayerNorm example above
   - Provide `with_scale()` to the user to make a promise about scale, overriding the above logic
