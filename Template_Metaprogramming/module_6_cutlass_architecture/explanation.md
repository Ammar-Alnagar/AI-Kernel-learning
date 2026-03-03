# Module 6: Introduction to CUTLASS Architecture

## Overview
This module introduces the CUTLASS 3.x architecture, focusing on its design philosophy, core components, and how it leverages template metaprogramming for high-performance linear algebra operations.

## Learning Objectives
By the end of this module, students will be able to:
- Understand CUTLASS 3.x architecture overview
- Grasp GEMM fundamentals and tile-based computation approach
- Identify and work with CUTLASS components (Threadblock, Warp, Instruction levels)
- Understand layout and stride concepts in CUTLASS
- Work with epilogues and fusion operations
- Navigate CUTLASS examples and source code

## Topic 1: CUTLASS 3.x Architecture Overview

CUTLASS (CUDA Templates for Linear Algebra Subroutines) is NVIDIA's collection of CUDA C++ template abstractions for implementing high-performance matrix multiplication (GEMM) and related computations.

### Core Philosophy
CUTLASS follows a hierarchical approach to matrix multiplication:
1. **Threadblock-level**: Each thread block handles a tile of the computation
2. **Warp-level**: Each warp processes a sub-tile within the thread block
3. **Instruction-level**: Individual CUDA instructions perform the actual math

```cpp
// Simplified CUTLASS-like structure
namespace cutlass_like {

// Core GEMM operation: D = alpha * A * B + beta * C
// Where C is the source accumulator and D is the destination
template<
    typename ElementA,        // Data type of operand A
    typename ElementB,        // Data type of operand B  
    typename ElementC,        // Data type of operand C/D
    typename LayoutA,         // Memory layout of A (row/column major)
    typename LayoutB,         // Memory layout of B
    typename LayoutC,         // Memory layout of C/D
    typename ElementAccumulator, // Accumulator data type
    int kAlignmentA = 128/sizeof_bits<ElementA>::value,  // Memory alignment
    int kAlignmentB = 128/sizeof_bits<ElementB>::value,
    int kAlignmentC = 128/sizeof_bits<ElementC>::value
>
struct GemmTraits {
    using ElementA = ElementA;
    using ElementB = ElementB;
    using ElementC = ElementC;
    using LayoutA = LayoutA;
    using LayoutB = LayoutB;
    using LayoutC = LayoutC;
    using ElementAccumulator = ElementAccumulator;
    
    static int const kAlignmentA = kAlignmentA;
    static int const kAlignmentB = kAlignmentB;
    static int const kAlignmentC = kAlignmentC;
};

} // namespace cutlass_like
```

### CUTLASS Component Hierarchy
```cpp
// CUTLASS follows a strict hierarchy of components
namespace cutlass_components {

// 1. Threadblock-level GEMM - orchestrates the entire tile computation
struct ThreadblockGemm {
    // Manages loading from global memory to shared memory
    // Coordinates warp-level operations
    // Handles storing results back to global memory
};

// 2. Warp-level GEMM - processes sub-tiles within threadblock
struct WarpGemm {
    // Uses warp-level primitives for efficient computation
    // Leverages tensor cores when available
    // Communicates with other warps in the threadblock
};

// 3. Instruction-level - actual math operations
struct InstructionGemm {
    // Maps to specific CUDA instructions (wmma, mma.sync, etc.)
    // Performs the fundamental multiply-add operations
};

} // namespace cutlass_components
```

## Topic 2: GEMM Fundamentals

### General Matrix Multiplication (GEMM)
The core operation in CUTLASS is GEMM: D = alpha * A * B + beta * C

```cpp
// Mathematical representation
/*
  A (M x K) * B (K x N) = C (M x N)
  D = alpha * A * B + beta * C
*/

// CUTLASS-style GEMM implementation outline
template<typename ElementA, typename ElementB, typename ElementC>
__global__ void simplified_gemm(
    ElementA const *A,
    ElementB const *B, 
    ElementC *C,
    ElementC *D,
    int M, int N, int K,
    ElementC alpha,
    ElementC beta) {
    
    // Thread's tile position in the matrix
    int block_row = blockIdx.y * 128;  // Example tile size
    int block_col = blockIdx.x * 128;
    
    // Each thread processes a sub-tile
    int thread_row = block_row + threadIdx.y * 8;  // Example sub-tile
    int thread_col = block_col + threadIdx.x * 8;
    
    // Accumulator for partial results
    ElementC accumulator[8][8] = {{ElementC(0)}};
    
    // Iterate through K dimension in tiles
    for (int k = 0; k < K; k += 32) {  // Example K tile size
        // Load fragments of A and B
        // Perform computation using tensor cores or regular math
        // Accumulate results
    }
    
    // Apply alpha and beta scaling
    for (int i = 0; i < 8; i++) {
        for (int j = 0; j < 8; j++) {
            int global_row = thread_row + i;
            int global_col = thread_col + j;
            
            if (global_row < M && global_col < N) {
                int idx = global_row * N + global_col;
                D[idx] = alpha * accumulator[i][j] + beta * C[idx];
            }
        }
    }
}
```

## Topic 3: Tile-Based Computation Approach

CUTLASS uses a tile-based approach to optimize memory access and computation patterns.

### Tiling Strategy
```cpp
// CUTLASS tiling parameters
struct TilingParameters {
    // Threadblock-level tile size
    static int const kBlockM = 128;  // Rows processed by one threadblock
    static int const kBlockN = 128;  // Columns processed by one threadblock
    static int const kBlockK = 32;   // Depth of tile (K dimension)
    
    // Warp-level tile size
    static int const kWarpM = 64;    // Rows processed by one warp
    static int const kWarpN = 64;    // Columns processed by one warp
    
    // Instruction-level tile size
    static int const kInstructionM = 16;  // Rows per MMA instruction
    static int const kInstructionN = 16;  // Cols per MMA instruction
    static int const kInstructionK = 16;  // Depth per MMA instruction (for FP16)
};
```

### Memory Layout and Tiling
```cpp
// Memory layout concepts in CUTLASS
namespace memory_layout {

// Row-major layout
struct RowMajor {
    __host__ __device__
    int operator()(int row, int col, int leading_dim) const {
        return row * leading_dim + col;  // Address = row * ld + col
    }
};

// Column-major layout
struct ColumnMajor {
    __host__ __device__
    int operator()(int row, int col, int leading_dim) const {
        return col * leading_dim + row;  // Address = col * ld + row
    }
};

// Blocked layout for cache optimization
template<int BlockHeight = 64, int BlockWidth = 64>
struct BlockedLayout {
    __host__ __device__
    int operator()(int row, int col, int leading_dim) const {
        int block_row = row / BlockHeight;
        int block_col = col / BlockWidth;
        int pos_in_block_row = row % BlockHeight;
        int pos_in_block_col = col % BlockWidth;
        
        // First address all blocks, then positions within block
        return (block_row * (leading_dim / BlockWidth) + block_col) * 
               (BlockHeight * BlockWidth) +
               (pos_in_block_row * BlockWidth + pos_in_block_col);
    }
};

} // namespace memory_layout
```

## Topic 4: CUTLASS Components

### Threadblock-Level Operations
```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>

// Example of CUTLASS threadblock-level GEMM
template<typename ElementA, typename ElementB, typename ElementC>
struct CutlassThreadblockGemm {
    using Gemm = cutlass::gemm::device::Gemm<
        ElementA, cutlass::layout::ColumnMajor,    // A matrix data type, layout
        ElementB, cutlass::layout::ColumnMajor,    // B matrix data type, layout
        ElementC, cutlass::layout::ColumnMajor,    // C/D matrix data type, layout
        ElementC                                     // Internal accumulator type
    >;
    
    Gemm gemm_operator;
    
    cutlass::Status operator()(
        cutlass::gemm::GemmCoord problem_size,
        ElementA alpha,
        cutlass::TensorRef<ElementA, cutlass::layout::ColumnMajor> ref_A,
        cutlass::TensorRef<ElementB, cutlass::layout::ColumnMajor> ref_B,
        ElementC beta,
        cutlass::TensorRef<ElementC, cutlass::layout::ColumnMajor> ref_C,
        cutlass::TensorRef<ElementC, cutlass::layout::ColumnMajor> ref_D,
        int split_k_slices = 1) {
        
        typename Gemm::Arguments args{
            problem_size,
            ref_A.non_const_ref(),
            ref_B.non_const_ref(), 
            ref_C.non_const_ref(),
            ref_D,
            {alpha, beta},
            split_k_slices
        };
        
        cutlass::Status status = gemm_operator(args);
        return status;
    }
};
```

### Warp-Level Operations
```cpp
// CUTLASS warp-level GEMM operations
namespace warp_level {

// Simplified warp-level GEMM fragment
template<typename Element, int M, int N>
struct WarpFragment {
    Element data[M][N];
    
    CUTLASS_DEVICE
    void load(Element const *ptr, int ldm) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < M; ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < N; ++j) {
                data[i][j] = ptr[i * ldm + j];
            }
        }
    }
    
    CUTLASS_DEVICE
    void store(Element *ptr, int ldm) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < M; ++i) {
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < N; ++j) {
                ptr[i * ldm + j] = data[i][j];
            }
        }
    }
};

} // namespace warp_level
```

### Instruction-Level Operations
```cpp
// CUTLASS instruction-level operations (simplified)
namespace instruction_level {

// MMA (Multiply-Accumulate) instruction wrapper
template<typename ElementA, typename ElementB, typename ElementC>
struct MmaInstruction {
    CUTLASS_DEVICE
    ElementC mma(ElementA a, ElementB b, ElementC c) {
        // On Tensor Core-capable hardware, this becomes a wmma instruction
        // On older hardware, this becomes regular FMA
        return a * b + c;
    }
    
    // Batched version for processing multiple elements
    template<int Elements>
    CUTLASS_DEVICE
    void mma_batch(
        ElementA const (&a)[Elements],
        ElementB const (&b)[Elements], 
        ElementC (&c)[Elements]) {
        
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < Elements; ++i) {
            c[i] = mma(a[i], b[i], c[i]);
        }
    }
};

} // namespace instruction_level
```

## Topic 5: Layout and Stride Concepts

### CUTLASS Layout Types
```cpp
// CUTLASS provides various layout types
namespace cutlass_layouts {

// Different memory layouts supported by CUTLASS
using ColumnMajor = cutlass::layout::ColumnMajor;
using RowMajor = cutlass::layout::RowMajor;
using ColumnMajorInterleavedK2 = cutlass::layout::ColumnMajorInterleavedK2;
using RowMajorInterleavedK2 = cutlass::layout::RowMajorInterleavedK2;

// Layout conversion utilities
template<typename Layout>
struct LayoutConverter {
    using cutlass_layout = Layout;
    
    CUTLASS_HOST_DEVICE
    static int get_index(int row, int col, int leading_dim) {
        cutlass_layout layout;
        return layout({row, col}, leading_dim);
    }
};

} // namespace cutlass_layouts
```

### Stride and Leading Dimension
```cpp
// Understanding strides in CUTLASS
struct StrideConcepts {
    /*
    For a matrix stored in memory:
    
    Row-major: A[i][j] stored at address A + i*ld + j
    Column-major: A[i][j] stored at address A + j*ld + i
    
    Where ld (leading dimension) is the distance between rows (row-major)
    or columns (column-major) in memory.
    */
    
    // Example: 3x3 matrix with leading dimension 5 (padded)
    // Row-major layout:
    // [0,0] [0,1] [0,2] [pad] [pad]
    // [1,0] [1,1] [1,2] [pad] [pad] 
    // [2,0] [2,1] [2,2] [pad] [pad]
    //
    // Address of A[i][j] = base_addr + i*5 + j
    // Leading dimension = 5 (not 3!)
};
```

## Topic 6: Epilogues and Fusion Operations

Epilogues in CUTLASS allow fusing additional operations with the main GEMM computation.

### Basic Epilogue
```cpp
// CUTLASS epilogue concepts
namespace epilogue_concepts {

// Simple epilogue: D = alpha * A * B + beta * C
struct LinearCombination {
    template<typename ElementC, typename ElementAccumulator>
    CUTLASS_DEVICE
    ElementC operator()(ElementAccumulator accumulator, ElementC source) {
        // This represents: D = alpha * accumulator + beta * source
        return alpha_ * ElementC(accumulator) + beta_ * source;
    }
    
    float alpha_;
    float beta_;
};

// Activation function epilogue: D = activation(alpha * A * B + beta * C)
template<typename ActivationFunctor>
struct LinearCombinationWithActivation {
    template<typename ElementC, typename ElementAccumulator>
    CUTLASS_DEVICE
    ElementC operator()(ElementAccumulator accumulator, ElementC source) {
        ElementC result = alpha_ * ElementC(accumulator) + beta_ * source;
        return activation_(result);
    }
    
    float alpha_;
    float beta_;
    ActivationFunctor activation_;
};

// Example activation functors
struct Relu {
    template<typename T>
    CUTLASS_DEVICE T operator()(T x) { return x > T(0) ? x : T(0); }
};

struct Sigmoid {
    template<typename T>
    CUTLASS_DEVICE T operator()(T x) { 
        return T(1) / (T(1) + exp(-x)); 
    }
};

} // namespace epilogue_concepts
```

### Advanced Epilogue with Broadcasting
```cpp
// Epilogue with bias addition (broadcasting)
namespace advanced_epilogue {

template<typename ElementOutput, typename ElementAccumulator>
struct LinearCombinationWithBias {
    ElementOutput alpha_;
    ElementOutput beta_;
    ElementOutput const *bias_ptr_;  // Bias vector
    
    CUTLASS_DEVICE
    ElementOutput operator()(
        ElementAccumulator accumulator, 
        ElementOutput source,
        int row, int col) {  // Position info for bias indexing
        
        ElementOutput intermediate = alpha_ * ElementOutput(accumulator) + 
                                   beta_ * source;
                                   
        if (bias_ptr_) {
            // Add bias for this row (broadcast along columns)
            intermediate += bias_ptr_[row];
        }
        
        return intermediate;
    }
};

} // namespace advanced_epilogue
```

## Hands-on Exercises

### Exercise 1: CUTLASS GEMM Instance
Create a CUTLASS GEMM instance with specific data types and layouts.

```cpp
// TODO: Create a CUTLASS GEMM instance for:
// - ElementA: half_t (FP16)
// - ElementB: half_t (FP16) 
// - ElementC/D: half_t (FP16)
// - Layout: Column-major for all matrices
// - Alpha: 1.0, Beta: 0.0
// Requirements:
// 1. Use appropriate CUTLASS template parameters
// 2. Set up proper tensor references
// 3. Configure problem size (M=1024, N=1024, K=1024)
```

### Exercise 2: Layout Conversion
Implement a function that converts between different memory layouts.

```cpp
// TODO: Implement a layout conversion function that:
// 1. Takes a matrix in row-major layout
// 2. Converts it to column-major layout
// 3. Uses CUTLASS layout types
// 4. Handles arbitrary matrix dimensions
```

### Exercise 3: Epilogue Implementation
Create a custom epilogue that implements a specific operation.

```cpp
// TODO: Create an epilogue that implements:
// D = alpha * A * B + beta * C + gamma * bias_vector
// Requirements:
// 1. Support bias addition per row
// 2. Use appropriate CUTLASS epilogue patterns
// 3. Handle different data types
// 4. Include proper error checking
```

## Solutions to Exercises

### Solution 1: CUTLASS GEMM Instance
```cpp
#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

// CUTLASS GEMM instance for FP16
using HalfGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                    // ElementA
    cutlass::layout::ColumnMajor,       // LayoutA
    cutlass::half_t,                    // ElementB
    cutlass::layout::ColumnMajor,       // LayoutB
    cutlass::half_t,                    // ElementC (and ElementD)
    cutlass::layout::ColumnMajor        // LayoutC/D
>;

void run_half_gemm() {
    const int M = 1024;
    const int N = 1024;
    const int K = 1024;
    
    // Allocate device memory
    cutlass::half_t *d_A, *d_B, *d_C, *d_D;
    cudaMalloc(&d_A, M * K * sizeof(cutlass::half_t));
    cudaMalloc(&d_B, K * N * sizeof(cutlass::half_t));
    cudaMalloc(&d_C, M * N * sizeof(cutlass::half_t));
    cudaMalloc(&d_D, M * N * sizeof(cutlass::half_t));
    
    // Create tensor references
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> ref_A(d_A, M);
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> ref_B(d_B, K);
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> ref_C(d_C, M);
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::ColumnMajor> ref_D(d_D, M);
    
    // Create GEMM operator
    HalfGemm gemm_operator;
    
    // Prepare arguments
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    typename HalfGemm::Arguments args{
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        {1.0f, 0.0f}  // alpha, beta
    };
    
    // Initialize the GEMM operator
    cutlass::Status status = gemm_operator.initialize(args);
    if (status != cutlass::Status::kSuccess) {
        // Handle error
        return;
    }
    
    // Run the GEMM
    status = gemm_operator();
    if (status != cutlass::Status::kSuccess) {
        // Handle error
        return;
    }
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_D);
}
```

### Solution 2: Layout Conversion
```cpp
#include <cutlass/layout/matrix.h>

template<typename Element>
void convert_layout(
    Element const *src,      // Source matrix
    Element *dst,            // Destination matrix  
    int rows, int cols,      // Matrix dimensions
    cutlass::layout::ColumnMajor src_layout,
    cutlass::layout::RowMajor dst_layout) {
    
    // Convert column-major to row-major
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            // Source index in column-major: j * rows + i
            int src_idx = src_layout({i, j}, rows);
            // Dest index in row-major: i * cols + j
            int dst_idx = dst_layout({i, j}, cols);
            
            dst[dst_idx] = src[src_idx];
        }
    }
}

// Generic layout converter
template<typename Element, typename LayoutSrc, typename LayoutDst>
void convert_layout_generic(
    Element const *src,
    Element *dst,
    int rows, int cols,
    LayoutSrc src_layout,
    LayoutDst dst_layout,
    int src_ld, int dst_ld) {
    
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            int src_idx = src_layout({i, j}, src_ld);
            int dst_idx = dst_layout({i, j}, dst_ld);
            dst[dst_idx] = src[src_idx];
        }
    }
}
```

### Solution 3: Epilogue Implementation
```cpp
#include <cutlass/epilogue/thread/linear_combination.h>

// Custom epilogue with bias addition
template<typename ElementOutput, typename ElementAccumulator>
struct LinearCombinationWithBiasAddition {
private:
    ElementOutput alpha_;
    ElementOutput beta_;
    ElementOutput gamma_;
    ElementOutput const *bias_ptr_;
    int row_stride_;

public:
    CUTLASS_HOST_DEVICE
    LinearCombinationWithBiasAddition(
        ElementOutput alpha = ElementOutput(1),
        ElementOutput beta = ElementOutput(0), 
        ElementOutput gamma = ElementOutput(1),
        ElementOutput const *bias_ptr = nullptr,
        int row_stride = 0)
        : alpha_(alpha), beta_(beta), gamma_(gamma), 
          bias_ptr_(bias_ptr), row_stride_(row_stride) {}

    /// Computes linear combination: D = alpha * accumulator + beta * source + gamma * bias
    CUTLASS_HOST_DEVICE
    ElementOutput operator()(
        ElementAccumulator accumulator, 
        ElementOutput source,
        int row, int column) const {
        
        ElementOutput intermediate = alpha_ * ElementOutput(accumulator) + 
                                   beta_ * source;
        
        if (bias_ptr_) {
            ElementOutput bias_val = bias_ptr_[row * row_stride_ + column % row_stride_];
            intermediate += gamma_ * bias_val;
        }
        
        return intermediate;
    }
};

// CUTLASS-compatible epilogue adapter
template<typename ElementOutput, typename ElementAccumulator>
struct CutlassCompatibleEpilogue {
    using FragmentAccumulator = cutlass::Array<ElementAccumulator, 4>;  // Example
    using FragmentSource = cutlass::Array<ElementOutput, 4>;
    using FragmentOutput = cutlass::Array<ElementOutput, 4>;
    
    LinearCombinationWithBiasAddition<ElementOutput, ElementAccumulator> operation;
    
    CUTLASS_HOST_DEVICE
    FragmentOutput operator()(
        FragmentAccumulator const &accumulator, 
        FragmentSource const &source,
        int row_start, int column_start) const {
        
        FragmentOutput output;
        for (int i = 0; i < FragmentOutput::kElements; ++i) {
            int row = row_start + (i / 2);      // Example mapping
            int col = column_start + (i % 2);   // Example mapping
            
            output[i] = operation(accumulator[i], source[i], row, col);
        }
        return output;
    }
};
```

## Advanced Topic: CUTLASS Template Patterns

Understanding CUTLASS's template-heavy architecture:

```cpp
// CUTLASS uses extensive template specialization
namespace cutlass_patterns {

// Example of CUTLASS's approach to algorithm selection
enum class GemmAlgorithm {
    kGemm,
    kSparse,
    kBatched,
    kGrouped
};

// Primary template
template<
    typename ElementA,
    typename LayoutA,
    typename ElementB, 
    typename LayoutB,
    typename ElementC,
    typename LayoutC,
    typename ElementAccumulator,
    GemmAlgorithm AlgorithmType = GemmAlgorithm::kGemm
>
struct GemmTraitsBase {
    using ElementA = ElementA;
    using LayoutA = LayoutA;
    using ElementB = ElementB;
    using LayoutB = LayoutB;
    using ElementC = ElementC;
    using LayoutC = LayoutC;
    using ElementAccumulator = ElementAccumulator;
    static GemmAlgorithm const kAlgorithm = AlgorithmType;
    
    // Default configurations
    static int const kThreadblockM = 128;
    static int const kThreadblockN = 128;
    static int const kThreadblockK = 32;
};

// Specialization for sparse GEMM
template<
    typename ElementA, typename LayoutA,
    typename ElementB, typename LayoutB, 
    typename ElementC, typename LayoutC,
    typename ElementAccumulator
>
struct GemmTraitsBase<ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC, 
                     ElementAccumulator, GemmAlgorithm::kSparse> {
    // Specialized parameters for sparse operations
    using ElementA = ElementA;
    using LayoutA = LayoutA;
    using ElementB = ElementB;
    using LayoutB = LayoutB;
    using ElementC = ElementC;
    using LayoutC = LayoutC;
    using ElementAccumulator = ElementAccumulator;
    static GemmAlgorithm const kAlgorithm = GemmAlgorithm::kSparse;
    
    // Different tiling for sparse operations
    static int const kThreadblockM = 128;
    static int const kThreadblockN = 128;
    static int const kThreadblockK = 16;  // Smaller K for sparsity
};

} // namespace cutlass_patterns
```

## Quiz Questions

1. What are the three levels of the CUTLASS computation hierarchy?

2. Explain the GEMM operation: D = alpha * A * B + beta * C

3. What is the purpose of tiling in CUTLASS and what are typical tile sizes?

4. What are CUTLASS epilogues and why are they important?

5. How do memory layouts (row-major vs column-major) affect GEMM performance?

## Summary
Module 6 introduced the CUTLASS 3.x architecture, covering its hierarchical design, GEMM fundamentals, tile-based computation approach, core components, layout concepts, and epilogue operations. This foundational understanding is essential for working with CUTLASS effectively.