# Project 01: Tiled GEMM — Putting It All Together

## Overview

This project combines Modules 1-6 into a complete tiled GEMM kernel. You will implement matrix multiplication with:
- TiledCopy for gmem→smem loading
- Swizzled shared memory for bank-conflict-free access
- TiledMMA for Tensor Core GEMM
- Double-buffered pipeline for load/compute overlap

## Job Mapping

**NVIDIA Deep Learning Software Engineer (Inference)** — "Design and implement custom CUDA kernels using CuTe for GEMM operations."

This is the exact pattern used in CUTLASS and TRT-LLM for GEMM kernels.

## Prerequisites

- Complete Modules 1-6
- Understand Tensor Core MMA atoms
- Comfortable with CuTe layouts and tensors

## Kernel Specification

| Parameter | Value |
|-----------|-------|
| Input A | [M, K] FP16 row-major |
| Input B | [K, N] FP16 column-major |
| Output C | [M, N] FP32 row-major |
| Tile size | [64, 64] with K=128 |
| smem layout | Swizzle<2, 3, 3> |
| Pipeline | 2-stage double buffer |

## Files

- `gemm.cu` — Complete tiled GEMM implementation

## Build and Run

```bash
nvcc -std=c++17 -I/path/to/cutlass/include \
     -arch=sm_89 -O3 \
     gemm.cu -o gemm && ./gemm
```

## Exit Criteria

Before moving to Project 02, you must:
1. Achieve >70% of peak Tensor TFLOPS for [1024, 1024] @ [1024, 1024]
2. Verify zero bank conflicts with Nsight Compute
3. Explain each pipeline stage (prologue/mainloop/epilogue)
4. Modify tile size and predict performance impact

## Roofline Analysis

This kernel is **compute-bound** for large matrices:
- Arithmetic intensity: O(M*N*K) / O(M*N + M*K + K*N) ≈ K/3
- For K=1024: ~340 FLOPs/byte (well above roofline knee)
- Expected: Near-peak Tensor TFLOPS

---

Next: **gemm.cu** — implement the full tiled GEMM.
