# Scale propagation proposal B

AutoScale relies on predicting the scale of tensors in computational graphs. Two key questions we have are:

 - What rules & approximations can we use to predict scale?
   - Therefore what is the design contract for a scaled op.
 - When should we requantise, in order to reset to an empirical scale?

In this proposal, we explore the two problems jointly, prioritising practical robustness over a more principled approach.

## High-level idea

1. For the first few iterations, run in higher-precision and calculate empirical "propagation scales" for each op, as well as collecting statistics for our rescaling strategy
2. These propagation scales are a multiplier which is applied to the input tensor-scales to get the output tensor-scale (_not_ simply the scale of the output tensor, see below for details)
3. After these initial iterations, a rescaling strategy will decide for each op how often it wishes to update its propagation scale (via empirical re-calculation), based on the statistics collected
4. More unstable ops (those where the tensor scale changes significantly) will be updated more often
5. The simplest version of the strategy is "on/off" - select a % of ops which will never be updated, and the rest update every `n` iterations
6. As the propagation scale is a _multiplier_, even when an op's propagation scale is fixed the benefits of any re-scalings elsewhere propagate
7. This is unlike Transformer Engine, where e.g. a drop in the scale of `grad(final_layer)` can require _every_ bwd-pass tensor to re-scale; in auto-scale only that op needs a new scale

## Design Principles

> quantization research is like printers. Nobody cares about printers. Nobody likes printers. But everybody is happy if printers do their job. --Tim Dettmers, 2022

- There should be as few assumptions as possible about the user's workload
- The default mode of operation should be very likely to give full accuracy on almost any model/workload, at the expense of speed
- There should be a clear, simple path presented to users for optimising to get the desired speedups, at the expense of accuracy
- There should be a reliable way of indicating to users if we might have over/underflowed
- The approach should be explainable in a few sentences

Plus, a key consideration: for big expensive runs, users don't have the opportunity to do a high-precision baseline to ensure auto-scale hasn't degraded things.
On this basis, we want to give users confidence that our method a) is fundamentally conservative, b) can flag up if it thinks it's seen a numerics issue.
Unlike unit scaling, autoscale users shouldn't have to cross their fingers and hope things turn out ok!

## Assumptions

We require the following assumptions for autoscale to work/be used effectively:
- the computational graph contains sub-graphs that are re-used (e.g. repeated layers, training loops)
- for re-used computational sub-graphs, no operation experiences a large, sudden shift in the way it propagates the scale of its inputs
- tensor dtypes that are sufficient for the first few iterations of a sub-graph are sufficient throughout (though an advanced scheme could in theory change dtypes)

Note that we do not need to assume:
- anything about the distributions of tensors
- anything about the type of workload - it need not be ML

## Usage

### (Contrived) Example

```python
x = torch.randn(batch_size, 16, dtype=torch.float32) * 3
linear = auto_scale.nn.Linear(16, 10, bias=False)
nn.init.normal_(linear.weight, std=5)

with auto_scale.scaling_context(analysis_iters=6, rescale_strategy=auto_scale.strategy.on_off(op_freq=1/3, loop_freq=1/20)):
    # in practice there would be a loop in here that runs at least `analysis_iters` times
    x = auto_scale.scale(x, dtype=torch.float8_e4m3)  # now a scaled float8 tensor, scale=3
    linear = auto_scale.scale(linear, dtype=torch.float8_e4m3)  # scaled float8, scale=5
    y = linear(x)  # y is also scaled float8, scale=3*sqrt(16)*5=60  # where the propagation scale is calculated empirically as sqrt(16)
    z = auto_scale.unscale(y, torch.float32)

print(x, x.type, x.scale)
print(linear.prop_scale)
print(y, y.type, y.scale)
print(z, z.dtype, z.std())
```
outputs:
```
tensor([...]) scaled_fp8 3
4  # Initially calculated empirically as `norm(linear.weight.value @ x.value)` (where `value` is the non-scale part of the scaled tensor), may then be re-calculated every n steps or frozen
tensor([...]) scaled_fp8 60
tensor([...]) float32 60
```

### Explanation

`with auto_scale.scaling_context(analysis_iters=6, rescale_strategy=auto_scale.strategy.on_off(op_freq=1/3, loop_freq=1/20))`

- the first 6 times scaled-tensor ops within this context are used, propagation scales will be calculated empirically in float32. In addition, statistics will be captured (e.g. abs max/min) for use by the rescaling strategy
- after 6 uses, the on/off rescaling strategy will be used - in this case 1/3 of the ops will be re-scaled every 20 steps, and the rest frozen (e.g. `scale`/`unscale` frozen, re-scale `linear`)
- the choice of which op to use is determined by some logic within the rescaling strategy. In this case it could just be based on how much the norm of the empirical propagation scale changes for each op over the 6 steps.
- other more complex rescaling strategies could be used to e.g. have different frequencies for different ops, or change those frequencies dynamically throughout training

`x = auto_scale.scale(x, dtype=torch.float8_e4m3)  # now a scaled float8 tensor, scale=3`

- the empirical scaling will set the tensor-scale here to the norm of the input tensor, and divide by that tensor-scale to get the tensor-value
- this is done in float32 for the first 6 iterations, then scaled float8
- if this op were frozen for subsequent operations, it would repeat this for every cast assuming the same tensor-scale(/norm)
- some work required to figure out how best to associate there "propagation scales" with their corresponding functions in software

`y = linear(x)  # y is also scaled float8, scale=3*sqrt(16)*5=60`

- this is a special implementation of a linear layer defined in the auto-scale library, designed to handle scaled tensors
- the logic is: `y.scale = x.scale * linear.weight.scale * linear.prop_scale`; `y.value = x.value @ linear.weight.value / linear.prop_scale`
- where the empirical calculation of the op's propagation scale is `linear.prop_scale = norm(y.value)` (done in fp32 for analysis phase, slightly more involved for re-scaling after that)
- the above example assumes the norm is std, but it could also be e.g. amax
- note that the transformer-engine way of doing this would be: `y.scale = linear.prop_scale`; `y.value = (x.scale * y.scale / linear.prop_scale) * x.value @ linear.weight.value`, `linear.prop_scale = y.scale * norm(y.value)`. This does not facilitate propagation of scale, which is the key difference.

## Over/underflow warnings

When we come to re-scale an op and the difference in old/new scales is sufficiently large that over/underflows may have caused a degradation, we should have some mechanism for flagging a warning to the user.
How we determine this requires some thought.
Of course, we aim to avoid this scenario - the frequency with which we decide to re-compute scales will be determined precisely to stop this happening. But if the user pushes their rescaling strategy too hard, then this should warn them.

The warnings should be reasonably conservative, with the hope that we can always signal to users if we've had a range issue.
For this reason we may also wish to implement a feature where for the final step of training _every_ op re-computes its scale (especially those which were frozen throughout), in order that they all check for the possibility that the model's final scales are inappropriate.

## Additional notes

- the hope here is that even though this method locks the user into having more empirical scale-calculations than a unit-scaling-like method, in practice only a small number of ops actually change the way they propagate scale significantly throughout training, and even these only need rescaling every e.g. 100 steps.
- if this is true, a sensible scaling stragety should be able to freeze most ops, and set the `loop_freq` to something fairly low. The overheads in such a system should be negligible.
- the default rescaling strategy hyperparameters provided to users should be conservative - for the on/off strategy we may simply set the default to be "on everywhere". The path to then getting speedups is then clear - start dropping the frequencies.
- this system is simpler than a unit-scaling-like method and has fewer assumptions & scary edge-cases
- most ops can be implemented in a very similar way to those described above
- this can be wrapped up into a graph transform, but similar to the unit scaling library it doesn't have to be. This way users have the options of e.g. just auto-scaling their matmuls and still getting the benefits. It's a less intrusive change.
