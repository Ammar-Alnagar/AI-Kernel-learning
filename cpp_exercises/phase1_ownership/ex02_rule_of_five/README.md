# ex02: Rule of Five

Implement all five special member functions for a resource-managing class.

## What You Build

A `ManagedBuffer` class with destructor, copy constructor, copy assignment, move constructor, and move assignment — each printing which method was called.

## What You Observe

Each operation (copy, move, assign) triggers the correct method. Moved-from objects are left empty but valid. All destructors fire cleanly at scope exit.

## CUTLASS/CUDA Mapping

CUTLASS kernel launchers use Rule of Five for device memory wrappers. The destructor calls `cudaFree`, move operations transfer device pointer ownership, copy operations either deep-copy (rare) or are deleted. This pattern appears in every `cutlass::DeviceAllocation` and `cutlass::TensorRef`.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex02 exercise.cpp && ./ex02
```
