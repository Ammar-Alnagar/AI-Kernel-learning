# CuTe Tutorials - Comprehensive Guide to CUDA Templates

Welcome to the **CuTe Tutorials**! This is a hands-on learning resource for mastering CuTe (CUDA Templates), the programming model used in CUTLASS 3.x for writing high-performance GPU kernels.

## 📚 Table of Contents

- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Setup and Build](#setup-and-build)
- [Tutorial Structure](#tutorial-structure)
- [Module Guide](#module-guide)
- [Running Examples](#running-examples)
- [Learning Path](#learning-path)
- [Additional Resources](#additional-resources)

## Overview

CuTe is a C++ template library that provides:
- **Layout Algebra**: Express memory layouts mathematically
- **Tensor Abstractions**: Work with multi-dimensional data
- **Tiled Copy**: Efficient memory transfer patterns
- **MMA Atoms**: Hardware-accelerated matrix operations
- **Shared Memory**: On-chip memory optimization
- **Software Pipelining**: Overlap memory and compute

These tutorials teach you CuTe through **runnable examples** and **hands-on exercises**.

## Prerequisites

### Hardware
- NVIDIA GPU with Compute Capability 7.0+ (Volta or newer)
- Recommended: Ampere (8.0+) for full feature support

### Software
- CUDA Toolkit 11.0 or later
- CMake 3.20 or later
- C++17 compatible compiler (GCC 7+, Clang 5+, MSVC 2017+)
- CUTLASS submodule initialized

### Required Knowledge
- Basic C++ programming
- Familiarity with CUDA concepts (threads, blocks, kernels)
- Understanding of matrix multiplication (GEMM)

## Setup and Build

### 1. Ensure CUTLASS Submodule is Initialized

```bash
cd /path/to/CuTE
git submodule update --init --recursive
```

### 2. Create Build Directory

```bash
cd CuTe_Tutorials
mkdir build && cd build
```

### 3. Configure with CMake

```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80;86;89"
```

Adjust `CMAKE_CUDA_ARCHITECTURES` for your GPU:
- RTX 30xx (Ampere): `80`
- RTX 40xx (Ada): `89`
- RTX 20xx (Turing): `75`
- V100 (Volta): `70`

### 4. Build All Tutorials

```bash
make -j$(nproc)
```

### 5. (Optional) Build Without Solutions

```bash
cmake .. -DBUILD_SOLUTIONS=OFF
make -j$(nproc)
```

## Tutorial Structure

```
CuTe_Tutorials/
├── 01_Intro_Layout_Algebra/
│   ├── examples/
│   │   ├── 01_layout_basics.cu
│   │   ├── 02_shape_and_stride.cu
│   │   └── 03_layout_composition.cu
│   ├── exercises/
│   │   └── exercise_01.cu
│   └── solutions/
│       └── exercise_01_solution.cu
├── 02_CuTe_Tensors/
│   ├── examples/
│   │   ├── 01_tensor_creation.cu
│   │   ├── 02_tensor_views.cu
│   │   └── 03_tensor_arithmetic.cu
│   ├── exercises/
│   │   └── exercise_02.cu
│   └── solutions/
│       └── exercise_02_solution.cu
├── 03_Tiled_Copy/
│   ├── examples/
│   │   ├── 01_copy_basics.cu
│   │   ├── 02_tiled_copy_2d.cu
│   │   └── 03_thread_mapping.cu
│   ├── exercises/
│   │   └── exercise_03.cu
│   └── solutions/
│       └── exercise_03_solution.cu
├── 04_MMA_Atoms/
│   ├── examples/
│   │   ├── 01_mma_introduction.cu
│   │   ├── 02_mma_shapes.cu
│   │   └── 03_mma_accumulation.cu
│   ├── exercises/
│   │   └── exercise_04.cu
│   └── solutions/
│       └── exercise_04_solution.cu
├── 05_Shared_Memory/
│   ├── examples/
│   │   ├── 01_shared_memory_basics.cu
│   │   ├── 02_swizzling.cu
│   │   └── 03_bank_conflict_free.cu
│   ├── exercises/
│   │   └── exercise_05.cu
│   └── solutions/
│       └── exercise_05_solution.cu
└── 06_Advanced_Patterns/
    ├── examples/
    │   ├── 01_gemm_pattern.cu
    │   └── 02_software_pipeline.cu
    └── solutions/
        └── (exercise solutions if applicable)
```

## Module Guide

### Module 01: Introduction to Layout Algebra

**Goal**: Understand how CuTe maps logical coordinates to memory offsets.

**Examples**:
- `tutorial_01_layout_basics`: Create row-major and column-major layouts
- `tutorial_01_shape_stride`: Understand shape and stride relationship
- `tutorial_01_composition`: Compose hierarchical layouts

**Key Concepts**:
- Layout = Shape + Stride
- Row-major vs Column-major
- Custom strides for padding
- Layout composition

**Exercise**: Create layouts with different strides and verify offset calculations.

### Module 02: CuTe Tensors

**Goal**: Learn to create and manipulate tensors in CuTe.

**Examples**:
- `tutorial_02_tensor_creation`: Create tensors from data
- `tutorial_02_tensor_views`: Slice and reshape tensors
- `tutorial_02_tensor_arithmetic`: Element-wise operations

**Key Concepts**:
- Tensor = Layout + Data pointer
- Zero-copy views
- Tensor slicing and dicing
- Arithmetic operations

**Exercise**: Create tensors, extract views, and perform operations.

### Module 03: Tiled Copy

**Goal**: Master efficient memory copy patterns.

**Examples**:
- `tutorial_03_copy_basics`: Introduction to tiled copy
- `tutorial_03_tiled_copy_2d`: 2D copy with thread blocks
- `tutorial_03_thread_mapping`: Map threads to data

**Key Concepts**:
- Tiled memory operations
- 2D thread organization
- Memory coalescing
- Vectorized loads

**Exercise**: Implement tiled copy kernels with different thread mappings.

### Module 04: MMA Atoms

**Goal**: Understand hardware-accelerated matrix operations.

**Examples**:
- `tutorial_04_mma_introduction`: MMA fundamentals
- `tutorial_04_mma_shapes`: Different MMA configurations
- `tutorial_04_mma_accumulation`: Accumulate over K-tiles

**Key Concepts**:
- D = A * B + C
- MMA shapes (16x8x16, 16x16x16, etc.)
- K-tile accumulation
- Tensor core utilization

**Exercise**: Calculate MMA requirements for different GEMM sizes.

### Module 05: Shared Memory

**Goal**: Optimize using shared memory and avoid bank conflicts.

**Examples**:
- `tutorial_05_shared_basics`: Shared memory fundamentals
- `tutorial_05_swizzling`: XOR-based conflict avoidance
- `tutorial_05_bank_conflict_free`: Design conflict-free layouts

**Key Concepts**:
- Shared memory banks
- Bank conflicts
- Padding strategies
- Swizzling techniques

**Exercise**: Design bank-conflict-free layouts for GEMM tiles.

### Module 06: Advanced Patterns

**Goal**: Combine all concepts into complete GEMM patterns.

**Examples**:
- `tutorial_06_gemm_pattern`: Complete GEMM structure
- `tutorial_06_software_pipeline`: Overlap memory and compute

**Key Concepts**:
- Thread block organization
- Shared memory tiling
- Software pipelining
- Producer-consumer pattern

## Running Examples

### Run a Single Example

```bash
cd build
./tutorial_01_layout_basics
./tutorial_02_tensor_creation
./tutorial_03_copy_basics
# ... and so on
```

### Run All Examples

```bash
cd build
for exe in tutorial_*; do echo "=== Running $exe ==="; ./$exe; done
```

### Run Exercise Solutions

```bash
cd build
./exercise_01_solution
./exercise_02_solution
# ... and so on
```

## Learning Path

### Beginner Path (Week 1-2)
1. Complete Module 01 (Layout Algebra)
2. Complete Module 02 (Tensors)
3. Do all exercises and compare with solutions

### Intermediate Path (Week 3-4)
1. Complete Module 03 (Tiled Copy)
2. Complete Module 04 (MMA Atoms)
3. Complete Module 05 (Shared Memory)
4. Do all exercises

### Advanced Path (Week 5-6)
1. Complete Module 06 (Advanced Patterns)
2. Study the complete GEMM pattern
3. Understand software pipelining
4. Experiment with different tile sizes

## Tips for Success

1. **Read the output**: Each example prints explanations along with results
2. **Modify and experiment**: Change parameters and observe effects
3. **Do the exercises**: Try before looking at solutions
4. **Use print()**: CuTe's `print()` function is invaluable for debugging
5. **Reference the docs**: Keep CUTLASS/CuTe documentation handy

## Common Issues

### Build Errors

**Problem**: `CUTLASS include directory not found`
**Solution**: Initialize the submodule: `git submodule update --init --recursive`

**Problem**: `CUDA architecture not supported`
**Solution**: Use appropriate `CMAKE_CUDA_ARCHITECTURES` for your GPU

### Runtime Errors

**Problem**: `CUDA error: invalid device function`
**Solution**: Rebuild with correct architecture for your GPU

**Problem**: Segmentation fault
**Solution**: Check that your GPU has sufficient compute capability

## Additional Resources

### Official Documentation
- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CuTe API Reference](https://nvidia.github.io/cutlass/)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

### Related Tutorials
- [NVIDIA CUTLASS Examples](https://github.com/NVIDIA/cutlass/tree/master/examples)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

### Books
- "Programming Massively Parallel Processors" - Hwu & Kirk
- "CUDA by Example" - Sanders & Kandrot

## Contributing

Found a bug or want to contribute improvements?
1. Fork the repository
2. Make your changes
3. Test thoroughly
4. Submit a pull request

## License

This tutorial series is provided under the MIT License. See the LICENSE file for details.

---

**Happy Learning!** 🚀

If you have questions or need help, please refer to the CUTLASS documentation or community forums.
