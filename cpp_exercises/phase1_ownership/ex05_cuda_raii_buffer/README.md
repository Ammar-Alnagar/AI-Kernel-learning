# ex05: CUDA RAII Buffer

Implement a RAII wrapper for CUDA device memory with move semantics.

## What You Build

A `DeviceBuffer` class that wraps `cudaMalloc`/`cudaFree`, with deleted copy operations and implemented move operations.

## What You Observe

Device memory is automatically freed when the `DeviceBuffer` goes out of scope. Moving transfers ownership without copying device memory. No manual `cudaFree` calls needed.

## CUTLASS/CUDA Mapping

This is the exact pattern CUTLASS uses for device memory management. `cutlass::DeviceAllocation` and similar types follow this RAII pattern. Every kernel launcher in CUTLASS uses move-only device wrappers to prevent leaks and double-frees.

## Build Command

```bash
nvcc -std=c++17 -O2 -arch=sm_89 -o ex05 exercise.cu && ./ex05
```
