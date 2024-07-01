# (WIP) Hopper GPU FP8 support & API

How does FP8 is exactly exposed in CUDA? And frameworks?

## Pytorch

https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815

Pytorch is exposing FP8 matmuls through the function [`_scaled_mm`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/native/cuda/Blas.cpp#L909). The main options exposed to this method:

* Output dtype: FP8 or higher.
* Fused bias.
* Scale of A, B, and output (the later only used if FP8 output?).
* `use_fast_accum`?

Under the hood, it is calling [`scaled_gemm`](https://github.com/pytorch/pytorch/blob/main/aten/src/ATen/cuda/CUDABlas.cpp#L1391), which is re-directing to CUDA `cublasltmatmul`.

Note: Pytorch is hard-setting `alpha_val=1.0` and `beta_val=0.0`. What is the difference between these on what you pass to the descriptor `CuBlasLtMatmulDescriptor`?

## JAX (and XLA/TensorFlow backend)

JAX supports mixed inputs FP8 matmuls (see [`dot_general`](https://github.com/google/jax/blob/main/jax/_src/lax/lax.py#L2929) lowering). Note that autograd rules have been softened to allow different floating point dtypes between forward activations and gradients (see JAX [commit](https://github.com/google/jax/commit/6f38f277b983c65086515f6e1062253514d5e544) and [Github ticket](https://github.com/google/jax/issues/18931)).

XLA compiler GEMM rewriter has various rules around FP8 matmuls to find common FP8 usage patterns, and issue optimal calls to `cublasltmatmul` (see TF XLA tests: https://github.com/tensorflow/tensorflow/blob/master/third_party/xla/xla/service/gpu/tests/gemm_rewrite_test.cc). In short, after rewriting, an HLO custom call is constructed. For instance:
```
 CHECK-NEXT:[[DOT_TUPLE:%[^ ]+]] = (<<F8E4M3>>[16,16]{1,0}, s8[{{[0-9]+}}]{0}) custom-call([[P0]], [[P1_TRANSPOSE]], [[C1]], [[C1]], [[C1]], /*index=5*/[[C1]]),
; CHECK:           custom_call_target="__cublas$lt$matmul$f8",
; CHECK:           backend_config={
; CHECK-DAG:         "alpha_real":1
; CHECK-DAG:         "alpha_imag":0
; CHECK-DAG:         "beta":0
; CHECK-DAG:         "dot_dimension_numbers":{
; CHECK-DAG:           "lhs_contracting_dimensions":["1"]
; CHECK-DAG:           "rhs_contracting_dimensions":["1"]
; CHECK-DAG:           "lhs_batch_dimensions":[]
; CHECK-DAG:           "rhs_batch_dimensions":[]
; CHECK-DAG:         }
; CHECK-DAG:         "precision_config":{
; CHECK-DAG:           "operand_precision":["DEFAULT","DEFAULT"]
; CHECK-DAG:         }
; CHECK-DAG:         "epilogue":"DEFAULT"
; CHECK:           }
```
where inputs, output scaling factors can be provided, as well as epilogue.

Note: fusing of output rescaling is support. See [TF/XLA PR 67206](https://github.com/tensorflow/tensorflow/pull/67206)


XLA RPC on FP8: https://github.com/openxla/xla/discussions/22

The main difference compared to Pytorch is XLA is attempting to check with pattern matching on the HLO graph if the output amax is computed,

The GEMM config proto definition has a boolean indicating whether to compute the `amax` or not: https://github.com/openxla/xla/blob/47e84570e9236156f669debd7a7c37b38d5456cf/xla/service/gpu/backend_configs.proto#L71

This values is set by the GEMM rewriter (based on the pattern matching result): https://github.com/openxla/xla/blob/47e84570e9236156f669debd7a7c37b38d5456cf/xla/service/gpu/gemm_rewriter.cc#L1443


Note: A, B, C, D scales are always allocated and passed to `cublasLtMatmul`, even when constant==1 (see https://github.com/openxla/xla/blob/main/xla/service/gpu/ir_emitter_unnested.cc#L744). Does it have an impact on performance??? 

`IsFastAccumEnabled` controls the accumulation algorithm used: https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/cuda_blas_lt.cc#L280. See Precision/Algorithm section: https://github.com/openxla/xla/blob/24b491c0ad8c00ab69421833376197ddd38fb2fb/xla/xla_data.proto#L948
Seems to be related to Triton `maxNumImpreciseAcc=32` setting and low level MMA_v3 dot instructions (see https://triton-lang.org/main/dialects/TritonNvidiaGPUOps.html#triton-nvidia-gpu-warp-group-dot-triton-nvidia-gpu-warpgroupdotop and https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-fragment-mma-884-f16).


Questions:

* Scale buffers are always allocated and passed to `cublasLtMatmul` (see https://github.com/openxla/xla/blob/main/xla/service/gpu/ir_emitter_unnested.cc#L744 and https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/cuda_blas_lt.cc#L400). Does it have an impact on performance? Or `cublasLtMatmul` is dynamically skipping the scaling step when `s=1`.

**Notes:**

* JAX supports [extended ML dtypes Array](https://github.com/google/jax/pull/20266) for jitted functions. Example of use: [scale dtypes](https://gist.github.com/mattjj/fd3b0a8c4f7533ddd9a56520d82871bf);


### Nvidia XLA-FP8

Nvidia release a `JAX-Toolbox` repository, with ["native" FP8 support](https://github.com/NVIDIA/JAX-Toolbox/blob/main/rosetta/docs/NATIVE_FP8.md) integrated in XLA. In short, it adds pattern matching to the XLA compiler to try to find and replace the classic quantize+matmul pattern, calling directly `cublasltmatmul` with proper input scales.

The expected semantics is a bit counter-intuitive, relying on a `bfloat16` cast and scale dequantization:
```python
@jax.jit
def matmul_fp8(a_fp8, a_scale, b_fp8, b_scale):
    # Dequantization: Up-cast from FP8 to a wider type.
    a = a_fp8.astype(jnp.bfloat16) * a_scale
    b = b_fp8.astype(jnp.bfloat16) * b_scale
    c = jax.lax.dot(a, b)
    return c
```

## Triton

Triton [FP8 tutorial](https://github.com/triton-lang/triton/blob/main/python/tutorials/09-persistent-matmul.py), with the following features:
* Use of Hopper TMA, with `create_2d_tma_descriptor`;
* `_experimental_descriptor_load/store`?
* Benchmark vs `CublasLt` directly (and Pytorch matmul);

## CUDA

The main API for scaled matmuls in CUDA is [`cublasltmatmul`](https://docs.nvidia.com/cuda/cublas/#cublasltmatmul), support the general computation:
$$ D = \alpha\cdot(A @ B) + \beta\cdot C $$
where $\alpha$ and $\beta$ are scales (FP32 scalars). Note `C` is not the bias term (there is an option for that).

The function has the following API:
```cpp
cublasStatus_t cublasLtMatmul(
      cublasLtHandle_t               lightHandle,
      cublasLtMatmulDesc_t           computeDesc,
      const void                    *alpha,
      const void                    *A,
      cublasLtMatrixLayout_t         Adesc,
      const void                    *B,
      cublasLtMatrixLayout_t         Bdesc,
      const void                    *beta,
      const void                    *C,
      cublasLtMatrixLayout_t         Cdesc,
      void                          *D,
      cublasLtMatrixLayout_t         Ddesc,
      const cublasLtMatmulAlgo_t    *algo,
      void                          *workspace,
      size_t                         workspaceSizeInBytes,
      cudaStream_t                   stream);
```
but additional options can be passed using `CuBlasLtMatmulDescriptor`:

* Inputs `A` and `B` scales, using `CUBLASLT_MATMUL_DESC_A_SCALE_POINTER` and `CUBLASLT_MATMUL_DESC_B_SCALE_POINTER`;
* Output `D` scale using `CUBLASLT_MATMUL_DESC_D_SCALE_POINTER`. (FIXME: how does compare to alpha?)
* `CUBLASLT_MATMUL_DESC_BIAS_POINTER`.
* `CUBLASLT_MATMUL_DESC_EPILOGUE`
* `CUBLASLT_MATMUL_DESC_AMAX_D_POINTER?` Only for FP8 result?
* `CUBLASLT_MATMUL_DESC_FAST_ACCUM`: fast accumulation in FP8 directly?
* [`cublasLtEpilogue_t`](https://docs.nvidia.com/cuda/cublas/#cublasltepilogue-t) fusing directly bias + activation (or its gradient).
* Finer grain block synchronization: https://docs.nvidia.com/cuda/cublas/#atomics-synchronization

Notes:
* Only `FP16` and `BF16` bias dtype is supported.

## Questions?

* Do people use FP8 outputs?
