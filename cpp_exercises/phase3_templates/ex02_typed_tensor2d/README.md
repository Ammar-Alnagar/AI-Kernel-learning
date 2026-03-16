# ex02: Typed Tensor2D Implementation

Implement a complete tensor class template with exported type aliases.

## What You Build

A `Tensor2D<T, IndexType>` class with type aliases (`value_type`, `index_type`, `pointer`), 2D access via `operator()`, and move semantics. Plus generic functions that query these types.

## What You Observe

Generic functions like `fill_sequential` work with any `Tensor2D` instantiation by querying `typename TensorType::value_type`. Move semantics transfer ownership without copying data.

## CUTLASS/CUDA Mapping

This is the foundation of CUTLASS tensor types. `cutlass::TensorRef` uses the same pattern. CUTLASS GEMM algorithms are templated on tensor types and query their properties via type aliases — enabling one implementation to work with float, half, int8, etc.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex02 exercise.cpp && ./ex02
```
