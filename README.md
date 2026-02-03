# CUTLASS 3.x & CuTe Comprehensive Learning Repository

This repository serves as a structured learning path for mastering CUTLASS 3.x and CuTe, focusing on composable abstractions for high-performance GPU programming on NVIDIA hardware. The learning path progresses from foundational concepts to advanced optimization techniques, building expertise incrementally.

## Target Hardware
- NVIDIA RTX 4060 (Compute Capability 8.9 / Ada Lovelace)

## Repository Structure

### Module 1: Layouts and Tensors (CuTe basics, nested layouts)
- Introduction to `cute::Layout` and `cute::Tensor`
- Understanding Shape and Stride algebra
- Composable tensor partitioning for thread mapping
- Mathematical representation of memory mappings

### Module 2: Tiled Copy (Vectorized global-to-shared memory movement)
- Efficient memory access patterns
- Vectorized loads and stores
- Shared memory tiling strategies
- Memory bandwidth optimization techniques

### Module 3: Tiled MMA (Using Tensor Cores via CuTe atoms)
- Tensor Core operations with CuTe
- MMA atom composition
- Fragment-based computation
- Performance optimization techniques

### Module 4: The Epilogue (Fused Bias-Add and ReLU implementations)
- Epilogue fusion techniques
- Memory-efficient activation functions
- Pipeline optimization
- Custom epilogue operations

### Module 5: Mainloop Pipelining - Temporal Overlap & Throughput
- Double-buffered approach for hiding memory latency
- Temporal overlap of load and compute operations
- Throughput optimization techniques
- High-performance kernel design principles

### Module 6: Fused Epilogues - Functional Avoiding VRAM Roundtrips
- Fusing bias-add and activation functions within GEMM kernels
- Eliminating intermediate memory accesses
- Memory efficiency through in-register operations
- Performance optimization for neural network inference

## Learning Flow Overview

The learning path follows a progressive approach where each module builds upon the previous ones:

1. **Foundation Building** (Modules 1-2): Establish core concepts of tensor layouts and efficient memory movement
2. **Computation Layer** (Module 3): Introduce Tensor Core operations and MMA
3. **Integration Layer** (Module 4): Combine computation with post-processing
4. **Optimization Layer** (Modules 5-6): Advanced techniques for performance maximization

## Prerequisites

- Solid understanding of CUDA programming fundamentals
- Familiarity with C++ templates and metaprogramming
- Basic knowledge of GPU memory hierarchy (global, shared, registers)
- Understanding of Tensor Core concepts
- Experience with matrix multiplication algorithms (GEMM)

## Setup Instructions

1. Clone the repository:
```bash
git clone https://github.com/Ammar-Alnagar/cutlass_learninig.git
```

2. Initialize submodules:
```bash
git submodule update --init --recursive
```

3. Build the project:
```bash
mkdir build && cd build
cmake ..
make -j$(nproc)
```

## Compilation Notes

Each module contains a standalone `main.cu` file that can be compiled individually using the provided NVCC command in the module's directory.

## Learning Methodology

This repository emphasizes "composable abstractions" over manual indexing. Instead of traditional nested loops, we focus on:

- **Mathematical Representation**: Thinking in terms of linear algebra and tensor operations
- **Functional Composition**: Building complex operations from simple, reusable components
- **Hardware Mapping**: Understanding how abstractions map to physical GPU resources
- **Performance Analysis**: Measuring and optimizing each component individually

### Key Principles

1. **Composability**: Each component should be reusable and combinable with others
2. **Abstraction**: Hide complexity behind clean, mathematically-grounded interfaces
3. **Performance**: Every abstraction should maintain or improve performance
4. **Correctness**: Mathematical precision and numerical accuracy are paramount

## Module Progression Strategy

### Sequential Learning
Follow the modules in order (1→2→3→4→5→6) to build knowledge incrementally.

### Hands-On Practice
Each module includes:
- Theoretical foundations
- Practical implementation exercises
- Performance measurement and analysis
- Debugging and troubleshooting techniques

### Integration Projects
After completing individual modules, integration projects combine concepts from multiple modules to solve complex problems.

## Performance Measurement Framework

Each module includes:
- Baseline performance metrics
- Optimization checkpoints
- Comparison with reference implementations
- Hardware-specific tuning parameters

## Troubleshooting and Debugging

Common issues and solutions are documented in the learning path, with emphasis on:
- Layout compatibility errors
- Memory access violations
- Performance bottlenecks
- Numerical precision issues

## Advanced Topics Covered

By the end of this learning path, you will understand:
- How to design high-performance GEMM kernels from scratch
- Techniques for maximizing Tensor Core utilization
- Advanced memory optimization strategies
- Pipeline design for latency hiding
- Fusion techniques for eliminating intermediate storage
- Hardware-specific optimization for Ada Lovelace architecture

## Expected Outcomes

Upon completing this learning path, you will be able to:
1. Design and implement high-performance GEMM kernels using CUTLASS 3.x
2. Apply composable abstractions to other computational problems
3. Optimize kernels for specific hardware targets
4. Analyze and debug performance bottlenecks in GPU code
5. Extend CUTLASS with custom operations and epilogues

## Additional Resources

- [Complete Learning Path Document](LEARNING_PATH.md)
- [CUTLASS 3.x Documentation](https://github.com/NVIDIA/cutlass)
- [CuTe Documentation](https://github.com/NVIDIA/cutlass/tree/master/tools/util/include/cute)
- [NVIDIA CUDA Programming Guide](https://docs.nvidia.com/cuda/)