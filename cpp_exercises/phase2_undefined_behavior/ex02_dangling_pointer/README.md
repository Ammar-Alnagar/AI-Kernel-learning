# ex02: Dangling Pointer (UB)

Debug undefined behavior caused by returning a pointer to freed memory.

## What You Build

A fix for a function that returns a dangling pointer by returning the RAII owner instead.

## What You Observe

The buggy version triggers sanitizer errors (heap-use-after-free) or prints garbage. The fixed version returns the `DataHolder` owner, which safely manages the memory lifetime via move semantics.

## CUTLASS/CUDA Mapping

Device memory functions should return RAII wrappers (`DeviceBuffer`), not raw `void*`. A raw pointer from `cudaMalloc` without an owner leads to leaks (no one frees) or use-after-free (freed by wrong owner). CUTLASS always returns ownership-transferring types.

## Build Command

```bash
# Buggy version (sanitizer catches UB)
g++ -std=c++20 -O2 -fsanitize=address -o ex02_buggy exercise.cpp && ./ex02_buggy

# Fixed version
g++ -std=c++20 -O2 -o ex02_fixed solution.cpp && ./ex02_fixed
```
