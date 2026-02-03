# CuTe Pedagogical Repository

## Overview
This repository contains 6 progressive modules to master CuTe (CUTLASS 3.x) for GPU kernel development, with focus on RTX 4060 (sm_89) architecture. Designed for engineers transitioning from manual CUDA indexing to compiler-friendly abstractions.

## Modules

### Module 01: Layout Algebra
- **Focus**: Shapes, Strides, and Hierarchical Layouts
- **Key Concepts**: Logical-to-physical memory mapping, debugging with cute::print()
- **Files**: README.md, layout_study.cu, BUILD.md

### Module 02: CuTe Tensors
- **Focus**: Wrapping pointers, Slicing, and Sub-tensors
- **Key Concepts**: Tensor creation, layout composition, memory access patterns

### Module 03: Tiled Copy
- **Focus**: Vectorized 128-bit loads and cp.async for sm_89
- **Key Concepts**: Coalesced memory access, async copy operations

### Module 04: MMA Atoms
- **Focus**: Direct Tensor Core access using hardware atoms
- **Key Concepts**: Matrix multiply-accumulate operations, WMMA instructions

### Module 05: Shared Memory & Swizzling
- **Focus**: Solving bank conflicts with Algebra
- **Key Concepts**: Shared memory optimization, swizzling patterns

### Module 06: Collective Mainloops
- **Focus**: Full producer-consumer pipeline
- **Key Concepts**: Complete kernel orchestration, thread cooperation

## Learning Path
Each module builds upon the previous, starting with fundamental layout concepts and progressing to complete kernel implementations. This progression mirrors how compilers like Mojo/MAX generate optimized kernels.

## Architecture Target
- **GPU**: NVIDIA RTX 4060 (sm_89)
- **CUDA**: 12.x with --expt-relaxed-constexpr
- **CUTLASS**: Version 3.x (CuTe library)

## Prerequisites
- Solid understanding of CUDA programming
- Familiarity with manual thread indexing (PMPP)
- Interest in compiler-generated optimizations