# Project 02: FlashAttention-2 Prefill — Capstone

## Overview

This is the capstone project combining all CuTe concepts into a complete FlashAttention-2 prefill kernel. You will implement:

- QK^T and PV GEMMs with TiledMMA
- TiledCopy with cp.async for K/V loading
- Swizzled shared memory layouts
- Double-buffered pipeline
- Online softmax with warp reductions
- Causal masking
- Full benchmarking vs. naive attention

## Job Mapping

This project directly maps to all target roles:

| Role | Requirement | How This Project Maps |
|------|-------------|----------------------|
| NVIDIA DL Software Engineer (Inference) | FlashAttention, TiledMMA, TiledCopy | Complete FlashAttention-2 implementation |
| NVIDIA DL Software Engineer (Optimization) | Kernel fusion, FP16/FP32 mixed precision | Fused attention with online softmax |
| Modular AI Kernel Engineer | High-performance attention, Tensor Cores | Optimized attention with swizzle + pipeline |
| Cerebras LLM Inference Performance | FlashAttention variants, profiling | Benchmarkable kernel with roofline analysis |
| Cerebras Inference ML Runtime | Latency/throughput optimization | Tokens/sec and latency metrics |

## Prerequisites

- Complete Modules 1-6
- Complete Project 01 (Tiled GEMM)
- Understand FlashAttention-2 algorithm

## Kernel Specification

| Parameter | Value |
|-----------|-------|
| Input Q | [batch, heads, seqlen, head_dim] FP16 |
| Input K | [batch, heads, seqlen, head_dim] FP16 |
| Input V | [batch, heads, seqlen, head_dim] FP16 |
| Output O | [batch, heads, seqlen, head_dim] FP32 |
| Row tile (Br) | 64 |
| Column tile (Bc) | 64 |
| smem layout | Swizzle<2, 3, 3> |
| Pipeline | 2-stage double buffer with cp.async |
| Masking | Causal (lower triangular) |

## Files

- `flash_attention.cu` — Complete FlashAttention-2 prefill implementation

## Build and Run

```bash
nvcc -std=c++17 -I/path/to/cutlass/include \
     -arch=sm_89 -O3 \
     flash_attention.cu -o flash_attention && ./flash_attention
```

## Exit Criteria

Before considering this complete:
1. Achieve >50% of theoretical attention FLOPS
2. Verify numerical correctness vs. naive attention (L2 error < 1e-4)
3. Benchmark tokens/sec and compare to naive implementation
4. Profile with Nsight and identify bottlenecks

## Roofline Analysis

FlashAttention is **memory-bandwidth-bound** for typical seqlen:
- Arithmetic intensity: O(seqlen² × head_dim) / O(seqlen × head_dim) = O(seqlen)
- For seqlen=1024: ~1024 FLOPs/byte (near roofline knee)
- Expected: 70-90% of peak memory bandwidth

The benchmark includes roofline analysis showing whether your kernel is compute or memory bound.

---

Next: **flash_attention.cu** — the capstone implementation.
