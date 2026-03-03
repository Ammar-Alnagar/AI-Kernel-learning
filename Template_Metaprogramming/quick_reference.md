# Quick Reference Guide: C++ Template Metaprogramming for CUTLASS 3.x

## Core Concepts

### Template Metaprogramming Fundamentals
- **Compile-time computation**: Use templates to perform calculations at compile time
- **Template recursion**: Implement loops and conditionals using recursive template instantiation
- **Type traits**: Use `<type_traits>` for compile-time type introspection
- **SFINAE**: Substitution Failure Is Not An Error - enable/disable templates based on type properties
- **Expression SFINAE**: Check if expressions are valid at compile time using `std::void_t`

### Essential Template Patterns
```cpp
// Enable-if pattern
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T> process(T val) {
    return val * 2;
}

// Tag dispatching
template<typename Iter>
void advance(Iter& it, int n, std::random_access_iterator_tag) {
    it += n;  // O(1) for random access iterators
}

// Policy-based design
template<typename MemoryPolicy, typename ThreadingPolicy>
class Container { /* ... */ };
```

### CUTLASS Template Conventions
- **Element types**: `ElementA`, `ElementB`, `ElementC` for data types
- **Layout types**: `LayoutA`, `LayoutB`, `LayoutC` for memory layouts
- **Architecture tags**: `arch::Sm80`, `arch::OpClassTensorOp` for hardware features
- **Shape templates**: `GemmShape<M, N, K>` for tile dimensions

## CUDA Fundamentals for CUTLASS

### Thread Hierarchy
```
Grid (multiple blocks)
├── Block (multiple threads, shared memory)
    ├── Warp (32 threads, executes in lockstep)
        └── Thread (single execution unit)
```

### Memory Types
- **Global memory**: Largest, slowest, accessible by all threads
- **Shared memory**: Faster, shared within block, limited size
- **Registers**: Fastest, private to each thread
- **Constant memory**: Cached, read-only, accessible by all threads

### Coalesced Access
- Consecutive threads should access consecutive memory addresses
- Row-major iteration for row-major data (good coalescing)
- Column-major iteration for column-major data (good coalescing)

## CUTLASS Architecture Components

### Three-Level Hierarchy
1. **Threadblock-level**: Each block handles a tile of the computation
2. **Warp-level**: Each warp processes a sub-tile within the block
3. **Instruction-level**: Individual CUDA instructions perform math

### Core GEMM Operation
```
D = alpha * A * B + beta * C
```
Where:
- A (M×K), B (K×N) are input matrices
- C (M×N) is the source accumulator
- D (M×N) is the destination
- alpha, beta are scaling factors

### Epilogue Operations
- **Linear combination**: `D = alpha * accumulator + beta * source`
- **Activation functions**: Apply ReLU, sigmoid, etc. in epilogue
- **Bias addition**: Add per-row or per-column biases
- **Clamping**: Limit output values to specific ranges

## Performance Optimization Techniques

### Memory Optimization
```cpp
// Add padding to avoid bank conflicts
template<int Rows, int Cols, typename Element>
struct PaddedSharedMemory {
    static constexpr int kPaddedCols = Cols + (Cols % 32 == 0 ? 1 : 0);
    Element storage[Rows][kPaddedCols];
};
```

### Occupancy Optimization
- Use `cudaOccupancyMaxPotentialBlockSize()` to find optimal block size
- Balance: enough threads to hide latency vs. resource limits
- Monitor: `theoretical_occupancy = active_blocks * block_size / max_threads_per_sm`

### Register Usage
- Use local arrays instead of many individual variables
- Control loop unrolling to manage register pressure
- Consider `__launch_bounds__(max_threads, min_blocks)` for hints

## Common CUTLASS Patterns

### Custom Epilogue Example
```cpp
template<typename ElementOutput, typename ElementAccumulator>
class LinearCombinationWithBias {
    // Custom epilogue with bias addition
    // D = alpha * A*B + beta * C + gamma * bias
};
```

### Quantized GEMM
```cpp
// INT8 inputs with INT32 accumulator
using QuantizedGemm = cutlass::gemm::device::Gemm<
    cutlass::int8_t, LayoutA,  // Input A
    cutlass::int8_t, LayoutB,  // Input B  
    int32_t, LayoutC,          // Output
    int32_t                    // Accumulator
>;
```

### Tensor Core Usage
```cpp
// For FP16 with FP32 accumulation on Tensor Core hardware
using TensorCoreGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t, cutlass::layout::ColumnMajor,  // A
    cutlass::half_t, cutlass::layout::ColumnMajor,  // B
    cutlass::half_t, cutlass::layout::ColumnMajor,  // C/D
    float,                                           // Accumulator
    cutlass::arch::OpClassTensorOp,                 // Operator class
    cutlass::arch::Sm80                            // Architecture
>;
```

## Profiling and Debugging

### Key Metrics to Monitor
- **Achieved occupancy**: Percentage of theoretical maximum occupancy
- **Memory bandwidth**: DRAM read/write throughput vs. peak
- **Tensor Core utilization**: For mixed-precision operations
- **SM efficiency**: Percentage of SMs actively working

### Common Issues and Solutions
- **Low occupancy**: Increase block size or reduce resource usage
- **Poor memory bandwidth**: Optimize access patterns for coalescing
- **Numerical errors**: Use higher precision accumulators or adjust tolerances
- **Compilation time**: Use precompiled kernels or reduce template complexity

## Integration Patterns

### PyTorch Integration
```cpp
torch::Tensor cutlass_gemm(torch::Tensor A, torch::Tensor B) {
    // Validate tensors
    // Create CUTLASS tensor refs
    // Launch CUTLASS kernel
    // Return result
}
```

### Performance Tuning
```cpp
// Auto-tuning framework
template<typename GemmOp>
auto find_optimal_config(problem_size) {
    // Test multiple configurations
    // Benchmark each
    // Return best performing config
}
```

## Key Headers to Remember
- `<cutlass/gemm/device/gemm.h>`: Basic GEMM operations
- `<cutlass/epilogue/thread/linear_combination.h>`: Epilogue operations
- `<cutlass/layout/matrix.h>`: Memory layout definitions
- `<cutlass/numeric_types.h>`: Data type definitions (half_t, etc.)
- `<cutlass/arch/mma.h>`: Hardware-specific operations

## Troubleshooting Quick Tips
- Use `static_assert` to catch template errors early
- Check alignment requirements for vectorized loads
- Verify memory layout compatibility between operations
- Profile with Nsight Compute to identify bottlenecks
- Test with smaller problem sizes first
- Validate numerics against reference implementations