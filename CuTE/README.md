# CuTe Learning Directory — LLM Inference Kernel Engineering

## Purpose

This is a **hands-on, code-first** CuTe curriculum for building production LLM inference kernels. You will write FlashAttention-2, tiled GEMM, and quantized kernels from scratch in CuTe/CUTLASS 3.x.

**Target roles:** NVIDIA (Inference Kernel Engineer), Cerebras (LLM Inference Performance), Modular (AI Kernel Engineer).

**Outcome:** In 6-8 weeks, you will have a public GitHub portfolio with runnable CuTe kernels that map directly to job requirements.

---

## Prerequisites

You already have:
- Expert C++ (memory, templates, performance)
- Expert CUDA (kernels, shared memory, Nsight profiling, roofline analysis)
- RTX 4060 (sm_89, Ada Lovelace) + Blackwell access
- CUDA 12.x + CUTLASS 3.x installed

You do **not** need:
- Linear algebra review
- Basic CUDA tutorials
- General GPU programming intro

---

## Learning Path

| Module | Topic | Outcome |
|--------|-------|---------|
| 01 | Layouts | Tile attention tensors, express GQA stride-0 broadcast |
| 02 | Tensors | Slice Q/K/V by head, iterate K/V blocks |
| 03 | TiledCopy | 128-bit vectorized loads, gmem→smem with cp.async |
| 04 | TiledMMA | Warp-level FP16 GEMM, fragment layout for softmax |
| 05 | Swizzle | Bank-conflict-free smem for MMA |
| 06 | Pipeline | Double-buffered FlashAttention-2 outer loop |
| Projects | Tiled GEMM, FlashAttention-2 | Production-ready kernels with benchmarks |

---

## How to Use This Directory

1. **Work sequentially.** Each module builds on the previous.
2. **Run every exercise.** All files compile and produce output.
3. **Answer CHECKPOINT questions** before moving to the next exercise.
4. **Profile with Nsight.** Each exercise includes `ncu` commands.
5. **Build the projects.** The capstone is FlashAttention-2 prefill.

---

## Environment Setup

```bash
# Verify CUDA
nvcc --version

# Verify CUTLASS 3.x is available (needed for CuTe headers)
# CuTe headers are in: <cutlass/include/cute>

# Compile an exercise
nvcc -std=c++17 -I/path/to/cutlass/include \
     -arch=sm_89 -O3 \
     ex01_basic_layouts.cu -o ex01 && ./ex01
```

---

## Job Mapping

Every exercise maps to a specific job requirement:

| Job | Requirement | Modules |
|-----|-------------|---------|
| NVIDIA DL Software Engineer (Inference) | CuTe kernels, FlashAttention, TiledMMA, TiledCopy | 03, 04, 06, Projects |
| NVIDIA DL Software Engineer (Model Optimization) | Kernel fusion, INT8/FP8 GEMM, TRT-LLM | 04, 05, 06 |
| Modular AI Kernel Engineer | High-performance attention/GEMM, Tensor Cores | 04, 05, 06, Projects |
| Cerebras LLM Inference Performance | FlashAttention variants, profiling-driven optimization | 03, 05, 06, Projects |
| Cerebras Inference ML Runtime | Latency/throughput optimization, vLLM/SGLang | 06, Projects |

---

## Exit Criteria (Full Directory)

Before considering this complete, you must be able to:

1. Write a tiled GEMM in CuTe with swizzled smem and double buffering
2. Write FlashAttention-2 prefill from scratch (no CUTLASS templates)
3. Profile with Nsight and identify bottlenecks (compute vs. memory bound)
4. Explain warp-level fragment layout for QK^T and PV GEMMs
5. Achieve >70% of roofline bandwidth on your tiled GEMM

---

## Next Steps

Start with **Module 01: Layouts**. It teaches the layout algebra you need to tile attention tensors and express GQA patterns.

```bash
cd CuTe/Module_01_Layouts
# Read README.md, then work through exercises in order
```
