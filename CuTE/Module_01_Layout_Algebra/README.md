# Module 01: Layout Algebra Fundamentals

## Overview
This module introduces CuTe's Layout Algebra - the mathematical foundation for expressing memory layouts in CUTLASS 3.x. Understanding layouts is crucial for transitioning from manual CUDA indexing (PMPP) to compiler-friendly abstractions.

## Key Concepts

### 1. Layout Definition
A `cute::Layout` is a mathematical mapping from logical coordinates to physical memory offsets:
```
offset = layout(logical_coord)
```

### 2. Shape and Stride
Every layout consists of:
- **Shape**: Dimensions of the logical space (e.g., `{M, N}`)
- **Stride**: Step sizes for traversing each dimension (e.g., `{stride_M, stride_N}`)

### 3. Memory Mapping Formula
For a 2D layout with shape `{M, N}` and strides `{stride_M, stride_N}`:
```
offset(i, j) = i * stride_M + j * stride_N
```

## Hierarchical Layouts
CuTe supports hierarchical layouts that compose multiple levels:
- **Tile Level**: High-level partitioning (e.g., thread blocks)
- **Thread Level**: Individual thread assignments
- **Element Level**: Vectorized access patterns

## Practical Benefits
- **Compiler Optimization**: Enables automatic vectorization and tiling
- **Portability**: Same code works across different architectures
- **Debugging**: Clear visualization of memory mappings with `cute::print()`
- **Composition**: Easy to combine and transform layouts

## Debugging with cute::print()
The `cute::print(layout)` function is invaluable for debugging memory-to-thread mappings:

```cpp
auto layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenRowMajor{});
cute::print(layout);  // Shows the layout structure and mappings
```

This outputs a human-readable representation of how logical coordinates map to memory offsets, making it easy to verify your layout construction is correct.

## Hands-on Exercise
Run `layout_study.cu` to visualize how different layouts map logical coordinates to memory offsets.