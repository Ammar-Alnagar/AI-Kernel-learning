# CUDA Fundamentals - Hands-On Exercises

This directory contains hands-on exercises designed to help you practice and master fundamental CUDA programming concepts. Each exercise focuses on a specific concept with incomplete code that you need to complete.

## Available Exercises

### 1. Vector Addition
- **File**: `vector_add_exercise.cu`
- **Concept**: Basic thread indexing and memory access
- **Task**: Complete the vector addition kernel (C = A + B)
- **Key Concepts**: Grid-block-thread hierarchy, bounds checking

### 2. Matrix Multiplication  
- **File**: `matrix_mul_exercise.cu`
- **Concept**: 2D thread indexing and nested loops
- **Task**: Complete the matrix multiplication kernel (C = A × B)
- **Key Concepts**: 2D indexing, matrix addressing, bounds checking

### 3. Reduction Operation
- **File**: `reduction_exercise.cu`
- **Concept**: Parallel reduction using shared memory
- **Task**: Complete the reduction kernel to compute array sum
- **Key Concepts**: Shared memory, thread synchronization, parallel algorithms

### 4. Memory Coalescing
- **File**: `memory_coalescing_exercise.cu`
- **Concept**: Optimizing memory access patterns
- **Task**: Fix memory access to ensure coalesced access
- **Key Concepts**: Memory access patterns, performance optimization

### 5. Shared Memory Banking
- **File**: `shared_memory_banking_exercise.cu`
- **Concept**: Avoiding bank conflicts in shared memory
- **Task**: Fix transpose operation to avoid bank conflicts
- **Key Concepts**: Shared memory banking, padding techniques

### 6. Atomic Operations
- **File**: `atomic_operations_exercise.cu`
- **Concept**: Handling race conditions with atomic operations
- **Task**: Complete atomic histogram computation
- **Key Concepts**: Race conditions, atomic functions, thread safety

### 7. CUDA Streams
- **File**: `cuda_streams_exercise.cu`
- **Concept**: Asynchronous execution with streams
- **Task**: Implement asynchronous memory transfers and kernel launches
- **Key Concepts**: CUDA streams, concurrency, overlapping operations

### 8. Warp-Level Primitives
- **File**: `warp_primitives_exercise.cu`
- **Concept**: Warp shuffle and vote operations
- **Task**: Complete kernels using warp-level primitives
- **Key Concepts**: Warp shuffles, vote operations, intra-warp communication

### 9. Master Tutorial
- **File**: `master_hands_on_tutorial.cu`
- **Concept**: Complete working examples of all concepts
- **Purpose**: Reference implementation showing all concepts working together

## How to Use These Exercises

1. **Start with the theory**: Review the fundamental concepts in the main tutorial files
2. **Attempt each exercise**: Open the `.cu` file and complete the missing code sections
3. **Check your work**: Use the corresponding `.md` file for hints and solutions
4. **Compile and test**: Use the Makefile to compile and run your implementations

## Compilation Instructions

Use the provided Makefile to compile individual exercises:

```bash
# Compile a specific exercise
nvcc vector_add_exercise.cu -o vector_add_exercise

# Or use the Makefile
make vector_add_exercise

# Run the exercise
./vector_add_exercise
```

## Learning Path

We recommend completing the exercises in this order:
1. Vector Addition
2. Matrix Multiplication
3. Memory Coalescing
4. Reduction Operation
5. Shared Memory Banking
6. Atomic Operations
7. Warp-Level Primitives
8. CUDA Streams

## Solutions

Complete solutions for reference are available in the master tutorial file. Try to solve each exercise on your own before consulting the solutions.

## Troubleshooting

If you encounter compilation errors:
- Make sure you're filling in the exact syntax required
- Check for missing semicolons, parentheses, or brackets
- Ensure variable names match exactly what's expected

If you encounter runtime errors:
- Verify bounds checking conditions
- Ensure proper memory allocation and deallocation
- Check for proper synchronization points