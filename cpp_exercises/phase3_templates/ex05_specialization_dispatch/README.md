# ex05: Template Specialization for Type Dispatch

Implement full template specializations for CUDA type-specific kernel configuration.

## What You Build

Specializations of `KernelConfig<T>` for `float`, `__half`, and `__nv_bfloat16`, each with different thread counts optimized for that type's characteristics.

## What You Observe

Each type gets its specialized configuration: float uses 256 threads, half/bfloat16 use 512 (less register pressure). The dispatch function automatically selects the right config via `KernelConfig<T>`.

## CUTLASS/CUDA Mapping

This is exactly how CUTLASS dispatches kernels. `cutlass::gemm::device::Gemm<float, ...>` is a full specialization with FP32-optimized kernels. `cutlass::gemm::device::Gemm<__half, ...>` uses FP16 tensor cores. Each specialization has different implementations.

## Build Command

```bash
nvcc -std=c++17 -O2 -arch=sm_89 -o ex05 exercise.cu && ./ex05
```
