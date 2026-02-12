# CUDA Fundamentals

This directory contains foundational tutorials for learning CUDA programming concepts. Each tutorial builds on fundamental GPU computing principles and provides both theoretical understanding and practical implementation.

## Available Tutorials

### 1. Thread Hierarchy
- **Concept**: Understanding the grid-block-thread organization in CUDA
- **Files**: 
  - `thread_hierarchy.md` - Theory and concepts
  - `thread_hierarchy_tutorial.cu` - Complete working examples
- **Topics Covered**:
  - Grid, block, and thread organization
  - Thread indexing variables (blockIdx, threadIdx, blockDim, gridDim)
  - 1D, 2D, and 3D thread configurations
  - Warp behavior and execution model
  - Grid-stride loops

### 2. Memory Hierarchy
- **Concept**: Understanding different memory types and their performance characteristics
- **Files**:
  - `memory_hierarchy.md` - Theory and concepts
  - `memory_hierarchy_tutorial.cu` - Complete working examples
- **Topics Covered**:
  - Global, shared, constant, and register memory
  - Memory coalescing patterns
  - Shared memory banking and conflicts
  - Memory access optimization strategies

### 3. Hands-On Kernels Practice
- **Concept**: Interactive exercises with incomplete code for students to complete
- **Files**:
  - `hands_on_kernels_tutorial.cu` - Original incomplete kernels to fill in
  - `hands_on_kernels_exercises.md` - Exercise instructions and guidance
  - Individual exercise files (see below)
- **Topics Covered**:
  - Vector addition implementation
  - Matrix multiplication
  - Reduction operations
  - Memory coalescing optimization
  - Shared memory banking fixes

### 4. Comprehensive Hands-On Exercises
- **Concept**: Separate files for each major CUDA concept with incomplete code
- **Files**:
  - `vector_add_exercise.cu` - Vector addition with missing indexing
  - `matrix_mul_exercise.cu` - Matrix multiplication with incomplete calculations
  - `reduction_exercise.cu` - Reduction operations with missing shared memory implementation
  - `memory_coalescing_exercise.cu` - Memory access pattern optimization
  - `shared_memory_banking_exercise.cu` - Shared memory banking conflict resolution
  - `atomic_operations_exercise.cu` - Atomic operations for race condition handling
  - `cuda_streams_exercise.cu` - Asynchronous execution with streams
  - `warp_primitives_exercise.cu` - Warp-level primitives usage
  - `master_hands_on_tutorial.cu` - Complete reference implementation
  - `HANDS_ON_EXERCISES.md` - Comprehensive guide to all exercises
- **Topics Covered**:
  - All fundamental CUDA programming patterns
  - Memory optimization techniques
  - Synchronization mechanisms
  - Advanced CUDA features

## Getting Started

1. Start with the theoretical concepts in the `.md` files
2. Study the complete examples in the `.cu` files
3. Practice with the hands-on exercises to solidify your understanding
4. Compile and run examples to see results:
   ```bash
   nvcc tutorial_file.cu -o tutorial_executable
   ./tutorial_executable
   ```

## Prerequisites

- Basic understanding of C/C++ programming
- Familiarity with parallel computing concepts
- CUDA-capable GPU with appropriate drivers installed

## Learning Path

1. Begin with thread hierarchy concepts and examples
2. Move to memory hierarchy concepts and examples
3. Practice with hands-on exercises to reinforce learning
4. Experiment with modifications to understand the concepts deeply