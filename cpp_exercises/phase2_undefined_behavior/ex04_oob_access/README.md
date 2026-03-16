# ex04: Out-of-Bounds Access (UB)

Debug undefined behavior caused by accessing array elements beyond valid bounds.

## What You Build

Fixes for stack and heap array accesses that go out of bounds, plus safe alternatives using standard containers.

## What You Observe

The buggy version triggers sanitizer errors (heap-buffer-overflow) and may corrupt adjacent variables. The fixed version uses proper bounds checking and safe containers with `.at()` that throw exceptions on OOB access.

## CUTLASS/CUDA Mapping

CUDA kernels must check thread indices before accessing shared/global memory. CUTLASS uses `if (threadIdx.x < TILE_SIZE)` guards in every kernel. OOB access in shared memory corrupts neighboring thread data, causing silent numerical errors.

## Build Command

```bash
# Buggy version (sanitizer catches UB)
g++ -std=c++20 -O2 -fsanitize=address -o ex04_buggy exercise.cpp && ./ex04_buggy

# Fixed version
g++ -std=c++20 -O2 -o ex04_fixed solution.cpp && ./ex04_fixed
```
