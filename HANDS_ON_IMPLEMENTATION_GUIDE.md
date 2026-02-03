# CUTLASS 3.x & CuTe Hands-On Implementation Guide

This guide provides practical, step-by-step instructions for implementing each module in the learning path, with code examples and exercises.

## Module 1: Layouts and Tensors - Hands-On

### Exercise 1.1: Basic Layout Creation
Create and manipulate simple layouts:

```cpp
#include <cute/layout.hpp>
#include <iostream>

void exercise_1_1() {
    // Create a simple 4x4 matrix layout
    auto layout = cute::make_layout(cute::make_shape(4, 4));
    
    std::cout << "Shape: " << layout.shape() << std::endl;
    std::cout << "Stride: " << layout.stride() << std::endl;
    
    // Access elements
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            int logical_idx = i * 4 + j;
            auto [phys_i, phys_j] = layout(logical_idx);
            std::cout << "(" << i << "," << j << ") -> (" << phys_i << "," << phys_j << ")" << std::endl;
        }
    }
}
```

### Exercise 1.2: Nested Layouts
Implement nested layouts for complex tensor operations:

```cpp
void exercise_1_2() {
    // Create a nested layout: 2x2 tiles of 4x4 matrices
    auto tile_layout = cute::make_layout(cute::make_shape(2, 2));
    auto element_layout = cute::make_layout(cute::make_shape(4, 4), cute::make_stride(1, 8));
    
    auto nested_layout = cute::composition(tile_layout, element_layout);
    
    std::cout << "Nested layout shape: " << nested_layout.shape() << std::endl;
    std::cout << "Nested layout stride: " << nested_layout.stride() << std::endl;
}
```

### Exercise 1.3: Custom Layout Design
Design a custom layout for a specific tensor access pattern:

```cpp
void exercise_1_3() {
    // Design a transposed layout
    auto row_major = cute::make_layout(cute::make_shape(8, 8), cute::make_stride(1, 8));
    auto col_major = cute::make_layout(cute::make_shape(8, 8), cute::make_stride(8, 1));
    
    // Verify the layouts work as expected
    std::cout << "Row major: " << row_major << std::endl;
    std::cout << "Column major: " << col_major << std::endl;
}
```

## Module 2: Tiled Copy - Hands-On

### Exercise 2.1: Basic Tiled Copy
Implement a simple tiled copy operation:

```cpp
#include <cute/tensor.hpp>
#include <cute/atom/copy_atom.hpp>

template<class Engine, class Layout>
__global__ void tiled_copy_kernel(cute::Tensor<Engine, Layout> global_tensor,
                                  cute::Tensor<Engine, Layout> shared_tensor) {
    // Create a copy atom for vectorized memory access
    auto copy_op = cute::make_copy(cute::Copy_Atom<cute::UniversalCopy<int>, int>{});
    
    // Partition tensors for threads
    auto thr_mma = cute::make_thread_slice(copy_op.get_layout_epilogue());
    
    // Perform the copy
    cute::copy(copy_op, global_tensor, shared_tensor);
}

void exercise_2_2() {
    // Allocate memory and launch kernel
    // This would typically involve CUDA memory allocation and kernel launch
}
```

### Exercise 2.2: Vectorized Memory Access
Implement vectorized load/store operations:

```cpp
void exercise_2_2() {
    // Define vectorized access patterns
    auto vector_layout = cute::make_layout(cute::make_shape(4), cute::make_stride(1));
    
    // This would involve creating tensors with appropriate strides
    // for vectorized memory access
}
```

## Module 3: Tiled MMA - Hands-On

### Exercise 3.1: Basic MMA Operation
Implement a simple GEMM using Tensor Cores:

```cpp
#include <cute/atom/mma_atom.hpp>

template<class ElementA, class ElementB, class ElementC>
__global__ void gemm_kernel(
    const ElementA* A,
    const ElementB* B,
    ElementC* C,
    int M, int N, int K) {
    
    // Define tensor layouts for operands
    auto A_layout = cute::make_fragment_like(cute::make_layout(cute::make_shape(M, K)));
    auto B_layout = cute::make_fragment_like(cute::make_layout(cute::make_shape(K, N)));
    auto C_layout = cute::make_fragment_like(cute::make_layout(cute::make_shape(M, N)));
    
    // Create MMA atom for Tensor Core operations
    auto mma_atom = cute::make_mma_atom<ElementC, ElementA, ElementB>();
    
    // Partition operands for threads
    auto thr_mma = cute::make_thread_slice(mma_atom.get_layout_MN());
    
    // Perform MMA operations
    // This would involve loading fragments and performing MMA
}
```

### Exercise 3.2: Fragment-Based Computation
Work with fragments for efficient computation:

```cpp
void exercise_3_2() {
    // Create fragments for computation
    // This would involve partitioning tensors into fragments
    // that can be processed by Tensor Cores
}
```

## Module 4: Epilogue - Hands-On

### Exercise 4.1: Fused Bias Addition
Implement a GEMM with fused bias addition:

```cpp
template<class ElementC, class ElementBias>
__device__ void fused_gemm_bias_relu(
    ElementC* C,
    const ElementBias* bias,
    int M, int N) {
    
    // Load bias values and add to C
    // Apply ReLU activation
    // Store results back to C
    
    // This would involve element-wise operations
    // fused with the main computation
}
```

### Exercise 4.2: Custom Epilogue Operations
Implement custom activation functions in the epilogue:

```cpp
// Example: Custom sigmoid activation in epilogue
template<class ElementC>
__device__ ElementC sigmoid(ElementC x) {
    return 1.0f / (1.0f + expf(-x));
}
```

## Module 5: Mainloop Pipelining - Hands-On

### Exercise 5.1: Double-Buffered Memory Access
Implement double buffering to hide memory latency:

```cpp
template<class ElementA, class ElementB, class ElementC>
__global__ void pipelined_gemm_kernel(
    const ElementA* A,
    const ElementB* B,
    ElementC* C,
    int M, int N, int K) {
    
    // Allocate shared memory for double buffering
    extern __shared__ char smem[];
    
    // Ping-pong buffers for A and B operands
    // Stage 1: Load data to buffer 1
    // Stage 2: Process data from buffer 1 while loading buffer 2
    // Stage 3: Process data from buffer 2 while loading buffer 1
    // Continue alternating
    
    // This creates overlapping of memory loads and computation
}
```

### Exercise 5.2: Pipeline Synchronization
Implement proper synchronization for pipelined operations:

```cpp
void exercise_5_2() {
    // Use CUDA cooperative groups for synchronization
    // Ensure proper ordering of pipeline stages
    // Handle edge cases where pipeline stages complete
}
```

## Module 6: Fused Epilogues - Hands-On

### Exercise 6.1: In-Register Operations
Implement operations that avoid intermediate memory access:

```cpp
template<class ElementC, class ElementBias>
__device__ void fused_compute_in_registers(
    cute::Tensor<ElementC> accumulator,
    const ElementBias* bias_ptr,
    int row_offset) {
    
    // Perform bias addition and activation in registers
    // without storing intermediate results to memory
    for (int i = 0; i < accumulator.size(); ++i) {
        accumulator(i) += bias_ptr[row_offset + i];
        accumulator(i) = fmaxf(0.0f, accumulator(i)); // ReLU
    }
}
```

### Exercise 6.2: Memory Traffic Reduction
Minimize memory traffic through fusion:

```cpp
void exercise_6_2() {
    // Combine multiple operations in a single kernel
    // Eliminate temporary storage requirements
    // Optimize for memory bandwidth constraints
}
```

## Integration Project: Complete Optimized GEMM

### Step 1: Design the Complete Kernel
Combine all modules into a single kernel:

```cpp
template<
    class ElementA,
    class ElementB, 
    class ElementC,
    class ElementAccumulator = float
>
__global__ void complete_optimized_gemm(
    const ElementA* A,
    const ElementB* B, 
    ElementC* C,
    const ElementC* bias,
    int M, int N, int K) {
    
    // 1. Layout design (Module 1)
    // 2. Tiled copy with vectorization (Module 2)
    // 3. Tensor Core MMA operations (Module 3)
    // 4. Fused epilogue with bias and activation (Module 4)
    // 5. Pipelined memory access (Module 5)
    // 6. In-register fusion (Module 6)
    
    // Implementation would integrate all techniques
}
```

### Step 2: Performance Tuning
Optimize the complete kernel:

```cpp
void performance_tuning() {
    // 1. Profile memory bandwidth utilization
    // 2. Measure Tensor Core utilization
    // 3. Adjust tile sizes for optimal occupancy
    // 4. Tune pipeline depth
    // 5. Optimize register usage
    // 6. Validate numerical accuracy
}
```

### Step 3: Validation and Testing
Test the complete implementation:

```cpp
bool validate_complete_gemm() {
    // Compare against reference implementation
    // Test various problem sizes
    // Verify numerical accuracy
    // Measure performance gains
    return true; // Placeholder
}
```

## Best Practices for Each Module

### Module 1 (Layouts):
- Always verify layout dimensions and strides
- Test boundary conditions
- Use mathematical notation to verify correctness

### Module 2 (Tiled Copy):
- Profile memory bandwidth
- Verify vectorization effectiveness
- Test different tile sizes

### Module 3 (MMA):
- Validate Tensor Core usage
- Check accumulator precision
- Test different data types

### Module 4 (Epilogue):
- Verify numerical accuracy after fusion
- Test memory access patterns
- Profile compute vs memory bound operations

### Module 5 (Pipelining):
- Ensure proper synchronization
- Test pipeline stability
- Measure latency hiding effectiveness

### Module 6 (Fusion):
- Validate memory traffic reduction
- Check register pressure
- Verify numerical accuracy

## Debugging Tips

### General Debugging:
- Use `printf` statements in kernels (with caution)
- Validate intermediate results
- Compare with simpler implementations
- Use CUDA debugging tools

### Performance Debugging:
- Use `nvprof` or `Nsight Compute` for profiling
- Monitor occupancy metrics
- Check memory bandwidth utilization
- Verify Tensor Core usage

### Correctness Debugging:
- Compare against CPU reference implementations
- Test with known inputs and expected outputs
- Verify boundary conditions
- Check for race conditions in shared memory

## Common Pitfalls and Solutions

### Layout Issues:
- **Pitfall**: Incorrect stride calculations
- **Solution**: Verify with mathematical formulas

### Memory Issues:
- **Pitfall**: Bank conflicts in shared memory
- **Solution**: Add padding or reorganize data layout

### Performance Issues:
- **Pitfall**: Low occupancy
- **Solution**: Reduce register usage or increase block size

### Numerical Issues:
- **Pitfall**: Precision loss in accumulators
- **Solution**: Use appropriate accumulator types

This hands-on guide provides practical implementation examples for each module in the learning path, helping you gain practical experience with CUTLASS 3.x and CuTe concepts.