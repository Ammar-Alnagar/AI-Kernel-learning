# CuTe Projects - Hands-on Kernel Implementation

## Overview

This module contains **hands-on projects** implementing real-world CUDA kernels using CuTe (CUTLASS 3.x). Each project builds upon concepts from the previous modules and guides you through implementing production-quality kernels step-by-step.

## Projects

### Core Projects (01-05)

| # | Project | Difficulty | Concepts |
|---|---------|------------|----------|
| 01 | [Vector Add](01_vector_add/) | ⭐ Beginner | CuTe tensors, element-wise ops, thread mapping |
| 02 | [GEMM](02_gemm/) | ⭐⭐ Intermediate | MMA atoms, tiling, shared memory |
| 03 | [Softmax](03_softmax/) | ⭐⭐ Intermediate | Reduction patterns, numerical stability |
| 04 | [FlashAttention](04_flash_attention/) | ⭐⭐⭐ Advanced | Tiled attention, online softmax, SRAM optimization |
| 05 | [FlashInfer](05_flashinfer/) | ⭐⭐⭐⭐ Expert | Variable sequences, page tables, PagedAttention |

### Advanced GEMM Projects (06-09)

| # | Project | Difficulty | Concepts |
|---|---------|------------|----------|
| 06 | [Tiled GEMM + Shared Memory](06_tiled_gemm_smem/) | ⭐⭐ Intermediate | Shared memory tiling, cooperative loading |
| 07 | [MMA GEMM + Tensor Cores](07_mma_gemm/) | ⭐⭐⭐ Advanced | Hardware MMA, WMMA, SM80 atoms |
| 08 | [Pipelined GEMM](08_pipelined_gemm/) | ⭐⭐⭐ Advanced | Async copy, software pipelining |
| 09 | [Vectorized Copy](09_vectorized_copy/) | ⭐ Beginner | float4 loads, 128-bit memory access |

### Quantized GEMM Projects (11-12)

| # | Project | Difficulty | Concepts |
|---|---------|------------|----------|
| 11 | [INT8 GEMM](11_int8_gemm/) | ⭐⭐⭐ Advanced | Quantized arithmetic, per-channel dequant |
| 12 | [FP8 GEMM](12_fp8_gemm/) | ⭐⭐⭐ Advanced | FP8 E4M3 format, mixed precision |

### Attention Projects (10, 13-14)

| # | Project | Difficulty | Concepts |
|---|---------|------------|----------|
| 10 | [Multi-Head Attention + KV-Cache](10_mha_kv_cache/) | ⭐⭐⭐ Advanced | MHA, incremental decoding, caching |
| 13 | [Fused GEMM + RoPE](13_gemm_rope/) | ⭐⭐⭐ Advanced | Rotary embeddings, fused operations |
| 14 | [MLA (Multi-head Latent Attention)](14_mla/) | ⭐⭐⭐⭐ Expert | Latent compression, memory-efficient attention |

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
# Core projects
make project_01_vector_add
make project_02_gemm
make project_03_softmax
make project_04_flash_attention
make project_05_flashinfer

# Advanced GEMM projects
make project_06_tiled_gemm_smem
make project_07_mma_gemm
make project_08_pipelined_gemm
make project_09_vectorized_copy

# Quantized GEMM projects
make project_11_int8_gemm
make project_12_fp8_gemm

# Attention projects
make project_10_mha_kv_cache
make project_13_gemm_rope
make project_14_mla
```

### Build a Single Project Standalone

Each project can be built **independently** from its own directory:

```bash
# Navigate to project directory
cd Projects/01_vector_add

# Create build directory
mkdir -p build && cd build

# Configure and build
cmake ..
make -j$(nproc)

# Run
./project_01_vector_add
```

This is useful for:
- Working on a single project without building everything
- Faster iteration during development
- Isolating build issues

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
