# CuTe Projects - Hands-on Kernel Implementation

## Overview

This module contains **hands-on projects** implementing real-world CUDA kernels using CuTe (CUTLASS 3.x). Each project builds upon concepts from the previous modules and guides you through implementing production-quality kernels step-by-step.

## Projects

| # | Project | Difficulty | Concepts |
|---|---------|------------|----------|
| 01 | [Vector Add](01_vector_add/) | ⭐ Beginner | CuTe tensors, element-wise ops, thread mapping |
| 02 | [GEMM](02_gemm/) | ⭐⭐ Intermediate | MMA atoms, tiling, shared memory |
| 03 | [Softmax](03_softmax/) | ⭐⭐ Intermediate | Reduction patterns, numerical stability |
| 04 | [FlashAttention](04_flash_attention/) | ⭐⭐⭐ Advanced | Tiled attention, online softmax, SRAM optimization |
| 05 | [FlashInfer](05_flashinfer/) | ⭐⭐⭐⭐ Expert | Variable sequences, page tables, PagedAttention |

## Building Projects

### Build All Projects

```bash
cd /path/to/CuTE
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Build Individual Project

Each project can be built individually:

```bash
# Build specific project
make project_01_vector_add
make project_02_gemm
make project_03_softmax
make project_04_flash_attention
make project_05_flashinfer
```

### Run a Project

```bash
./project_01_vector_add
./project_02_gemm
# etc...
```

## How to Use This Module

### Step-by-Step Approach

1. **Read the README** in each project directory first
2. **Study the code comments** - they contain guided exercises
3. **Complete the TODO sections** - marked with `// TODO:` comments
4. **Build and test** your implementation
5. **Compare with reference** - solutions are in `solution/` subdirectory

### Learning Philosophy

- **Minimal scaffolding**: Code provides structure, you implement the core logic
- **Progressive difficulty**: Each project adds new complexity
- **Real-world patterns**: Implementations mirror production libraries
- **Performance awareness**: Comments explain optimization trade-offs

## Prerequisites

Before starting these projects, ensure you have completed:

- ✅ Module 01: Layout Algebra (Shapes, Strides, Layouts)
- ✅ Module 02: CuTe Tensors (Tensor creation, slicing)
- ✅ Module 03: Tiled Copy (Vectorized loads)
- ✅ Module 04: MMA Atoms (Tensor Core operations)
- ✅ Module 05: Shared Memory Swizzling (Bank conflict avoidance)
- ✅ Module 06: Collective Mainloops (Producer-consumer pipelines)

## Architecture Target

- **GPU**: NVIDIA RTX 4060 (sm_89, Ada Lovelace)
- **CUDA**: 12.x
- **CUTLASS**: 3.x (CuTe library)

## Project Structure

Each project directory contains:

```
project_name/
├── README.md           # Detailed walkthrough and theory
├── project_name.cu     # Main implementation (with TODOs)
├── CMakeLists.txt      # Individual build configuration
├── solution/           # Reference implementation
│   └── project_name_solution.cu
└── tests/              # Verification tests
    └── test_project_name.cu
```

## Tips for Success

1. **Start simple**: Get a working implementation first, optimize later
2. **Use print debugging**: `cute::print()` is your friend
3. **Verify correctness**: Always compare against CPU reference
4. **Profile early**: Use `nvprof` or Nsight Compute to find bottlenecks
5. **Read the docs**: CuTe header files contain valuable examples

## Additional Resources

- [CUTLASS Documentation](https://github.com/NVIDIA/cutlass)
- [CuTe Tutorial](https://github.com/NVIDIA/cutlass/tree/main/media/docs/cute)
- [FlashAttention Paper](https://arxiv.org/abs/2205.14135)
- [FlashInfer Paper](https://arxiv.org/abs/2401.14238)

## Next Steps

After completing all projects:

1. Experiment with different tile sizes
2. Try targeting other GPU architectures (sm_80, sm_90)
3. Implement fused kernels (e.g., GEMM + Bias + ReLU)
4. Explore multi-block cooperation
5. Study the official CUTLASS examples

---

**Ready to build high-performance kernels? Start with [Project 01](01_vector_add/)!**
