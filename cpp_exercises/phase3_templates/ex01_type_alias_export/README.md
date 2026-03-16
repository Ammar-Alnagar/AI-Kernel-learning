# ex01: Type Aliases in Templates

Learn the `using value_type = T` pattern that CUTLASS uses everywhere.

## What You Build

A `Tensor2D<T>` class template that exports its element type via `using value_type = T`, plus functions that query this exported type.

## What You Observe

The exported type can be queried from outside using `typename Tensor2D<float>::value_type`. Type aliases like `FloatTensor = Tensor2D<float>` create convenient shorthands.

## CUTLASS/CUDA Mapping

This is the exact pattern CUTLASS uses: `cutlass::TensorRef<T, Layout>::value_type`, `cutlass::gemm::GemmCoord::IndexType`. Every policy class exports its configuration via type aliases. Reading CUTLASS requires fluency in this pattern.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex01 exercise.cpp && ./ex01
```
