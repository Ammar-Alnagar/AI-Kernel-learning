# Module 7: CUTLASS Template Patterns and Idioms

## Overview
This module explores the specific template patterns and idioms used in CUTLASS 3.x, focusing on how the library leverages advanced C++ template metaprogramming techniques for high-performance GPU computing.

## Learning Objectives
By the end of this module, students will be able to:
- Understand CUTLASS template parameter conventions
- Recognize and implement dispatch patterns used in CUTLASS
- Apply template specialization strategies similar to those in CUTLASS
- Work with hardware-specific optimizations in templates
- Integrate CUTLASS math instructions effectively
- Optimize memory access patterns using CUTLASS approaches

## Topic 1: CUTLASS Template Parameter Conventions

CUTLASS uses a consistent and well-defined convention for template parameters that enables flexible and efficient code generation.

### Standard Template Parameter Order
```cpp
// Typical CUTLASS template parameter pattern
template<
    typename ElementA,           // Data type of operand A
    typename LayoutA,            // Memory layout of A
    typename ElementB,           // Data type of operand B
    typename LayoutB,            // Memory layout of B
    typename ElementC,           // Data type of operand C/D
    typename LayoutC,            // Memory layout of C/D
    typename ElementAccumulator, // Internal accumulator type
    typename OperatorClass,      // Operator class (simt, tensorop, etc.)
    typename ArchTag,            // Architecture tag (compute_70, compute_80, etc.)
    typename ThreadblockShape,   // Threadblock-level tile shape
    typename WarpShape,          // Warp-level tile shape
    typename InstructionShape,   // Instruction-level tile shape
    typename EpilogueOutputOp,   // Epilogue operation
    typename ThreadblockSwizzle, // Threadblock swizzling function
    int Stages,                  // Number of pipeline stages
    int AlignmentA,              // Memory alignment for A
    int AlignmentB,              // Memory alignment for B
    int AlignmentC               // Memory alignment for C
>
struct CutlassGemmTemplate {
    // Implementation details...
};
```

### Example: Concrete CUTLASS Template Instantiation
```cpp
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>
#include <cutlass/layout/matrix.h>

// FP16 GEMM for Tensor Cores on Ampere architecture
using CutlassFp16TensorOpGemm = cutlass::gemm::device::Gemm<
    cutlass::half_t,                           // ElementA
    cutlass::layout::ColumnMajor,              // LayoutA
    cutlass::half_t,                           // ElementB
    cutlass::layout::ColumnMajor,              // LayoutB
    cutlass::half_t,                           // ElementC
    cutlass::layout::ColumnMajor,              // LayoutC
    cutlass::half_t,                           // ElementAccumulator
    cutlass::arch::OpClassTensorOp,            // OperatorClass
    cutlass::arch::Sm80,                       // ArchTag
    cutlass::gemm::GemmShape<128, 128, 32>,   // ThreadblockShape
    cutlass::gemm::GemmShape<64, 64, 32>,     // WarpShape
    cutlass::gemm::GemmShape<16, 8, 16>,      // InstructionShape
    cutlass::epilogue::thread::LinearCombination<  // EpilogueOutputOp
        cutlass::half_t, 128 / cutlass::sizeof_bits<cutlass::half_t>::value
    >,
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>, // Swizzle
    3,                                         // Stages
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, // AlignmentA
    128 / cutlass::sizeof_bits<cutlass::half_t>::value, // AlignmentB
    128 / cutlass::sizeof_bits<cutlass::half_t>::value  // AlignmentC
>;
```

## Topic 2: Dispatch Patterns in CUTLASS

CUTLASS uses sophisticated dispatch patterns to select optimal implementations based on compile-time and runtime parameters.

### Tag-Based Dispatch
```cpp
// CUTLASS-style tag dispatching
struct TensorOpClass {};
struct SimtClass {};
struct WmmaTensorOpClass {};

// Primary dispatcher
template<typename OperatorClass>
struct GemmDispatch;

// Specialization for Tensor Cores
template<>
struct GemmDispatch<TensorOpClass> {
    template<typename ElementA, typename ElementB, typename ElementC>
    using GemmKernel = TensorOpGemmKernel<ElementA, ElementB, ElementC>;
};

// Specialization for SIMT
template<>
struct GemmDispatch<SimtClass> {
    template<typename ElementA, typename ElementB, typename ElementC>
    using GemmKernel = SimtGemmKernel<ElementA, ElementB, ElementC>;
};

// Usage
template<typename ElementA, typename ElementB, typename ElementC, typename OpClass>
void launch_gemm() {
    using Kernel = typename GemmDispatch<OpClass>::template GemmKernel<ElementA, ElementB, ElementC>;
    // Launch Kernel...
}
```

### Architecture-Specific Dispatch
```cpp
// Architecture tags
struct Sm70 {};
struct Sm75 {};
struct Sm80 {};
struct Sm90 {};

// Architecture dispatcher
template<typename ArchTag>
struct ArchDispatch;

template<>
struct ArchDispatch<Sm80> {
    static constexpr bool has_tensor_cores = true;
    static constexpr int tensor_core_size = 16;  // For FP16
    
    template<typename Element>
    using MainLoop = TensorCoreMainLoop<Element>;
};

template<>
struct ArchDispatch<Sm70> {
    static constexpr bool has_tensor_cores = false;
    static constexpr int tensor_core_size = 0;
    
    template<typename Element>
    using MainLoop = SimtMainLoop<Element>;
};

// Conditional compilation based on architecture
template<typename ArchTag>
struct ConditionalFeatures {
    using arch_dispatch = ArchDispatch<ArchTag>;
    
    template<typename T>
    static CUTLASS_DEVICE void compute(T& result, T a, T b) {
        if constexpr (arch_dispatch::has_tensor_cores) {
            // Use tensor core instructions
            result = tensor_core_multiply_add(a, b, result);
        } else {
            // Use regular math
            result = a * b + result;
        }
    }
};
```

## Topic 3: Template Specialization Strategies in CUTLASS

CUTLASS extensively uses template specialization to provide optimized implementations for specific cases.

### Partial Specialization for Data Types
```cpp
// CUTLASS-style type specialization
template<typename Element>
struct NumericConverter {
    CUTLASS_HOST_DEVICE
    Element operator()(Element src) {
        return src;  // Default identity conversion
    }
};

// Specialization for converting to half
template<>
struct NumericConverter<cutlass::half_t> {
    CUTLASS_HOST_DEVICE
    cutlass::half_t operator()(float src) {
        return cutlass::half_t(src);
    }
};

// Specialization for converting from half
template<>
struct NumericConverter<float> {
    CUTLASS_HOST_DEVICE
    float operator()(cutlass::half_t src) {
        return float(src);
    }
};

// Specialization for integer quantization
template<>
struct NumericConverter<int8_t> {
    CUTLASS_HOST_DEVICE
    int8_t operator()(float src) {
        return static_cast<int8_t>(round(src));
    }
};
```

### Shape-Based Specialization
```cpp
// CUTLASS-style shape specialization
template<int M, int N, int K>
struct GemmShape {
    static int const kM = M;
    static int const kN = N;
    static int const kK = K;
};

// Specialization for common shapes
template<>
struct GemmShape<128, 128, 32> {
    static int const kM = 128;
    static int const kN = 128;
    static int const kK = 32;
    
    // Optimized parameters for this specific shape
    static int const kOptimalStages = 3;
    static int const kPreferredAligment = 128;
};

// Shape traits for optimization
template<typename Shape>
struct ShapeTraits {
    static bool const kIsPowerOfTwo = 
        (Shape::kM & (Shape::kM - 1)) == 0 &&
        (Shape::kN & (Shape::kN - 1)) == 0 &&
        (Shape::kK & (Shape::kK - 1)) == 0;
    
    static int const kArea = Shape::kM * Shape::kN;
    static int const kVolume = Shape::kM * Shape::kN * Shape::kK;
};
```

## Topic 4: Hardware-Specific Optimizations

CUTLASS incorporates hardware-specific optimizations through template parameters and specializations.

### Architecture Feature Detection
```cpp
// Hardware feature detection using templates
template<typename ArchTag>
struct ArchTraits {
    static bool const kSupportsTensorOps = false;
    static bool const kSupportsWmma = false;
    static int const kWarpSize = 32;
    static int const kMaxSharedMemory = 48 * 1024;  // 48KB default
};

// Specialization for Turing architecture
template<>
struct ArchTraits<cutlass::arch::Sm75> {
    static bool const kSupportsTensorOps = true;
    static bool const kSupportsWmma = true;
    static int const kWarpSize = 32;
    static int const kMaxSharedMemory = 64 * 1024;  // 64KB on Turing+
};

// Specialization for Ampere architecture
template<>
struct ArchTraits<cutlass::arch::Sm80> {
    static bool const kSupportsTensorOps = true;
    static bool const kSupportsWmma = false;  // Replaced by new MMA instructions
    static int const kWarpSize = 32;
    static int const kMaxSharedMemory = 164 * 1024;  // 164KB on Ampere
};
```

### Tensor Core Support
```cpp
// Tensor core operation wrapper
template<typename ElementA, typename ElementB, typename ElementC, typename ArchTag>
struct TensorOp {
    CUTLASS_DEVICE
    static void mma(ElementC &d, ElementA a, ElementB b, ElementC c) {
        // Default implementation (non-tensor core)
        d = a * b + c;
    }
};

// Specialization for architectures with tensor cores
template<>
struct TensorOp<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::arch::Sm80> {
    CUTLASS_DEVICE
    static void mma(cutlass::half_t &d, cutlass::half_t a, cutlass::half_t b, cutlass::half_t c) {
        // Use actual tensor core instruction on Sm80
        asm volatile(
            "mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0}, {%1}, {%2}, {%3};"
            : "=h"(reinterpret_cast<cutlass::uint16_t&>(d))
            : "h"(reinterpret_cast<cutlass::uint16_t const&>(a)),
              "r"(reinterpret_cast<cutlass::uint32_t const&>(b)),
              "h"(reinterpret_cast<cutlass::uint16_t const&>(c))
        );
    }
};
```

## Topic 5: CUTLASS Math Instructions Integration

CUTLASS abstracts hardware-specific math instructions through template interfaces.

### Math Instruction Abstraction
```cpp
// Abstract math instruction interface
template<typename ElementA, typename ElementB, typename ElementC, typename ArchTag>
struct MmaOperator {
    CUTLASS_DEVICE
    void operator()(
        Array<ElementC, 2> &d,           // Destination
        Array<ElementA, 2> const &a,     // Operand A
        Array<ElementB, 1> const &b,     // Operand B
        Array<ElementC, 2> const &c      // Source accumulator
    ) {
        // Default implementation using regular math
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < 2; ++i) {
            d[i] = a[i] * b[0] + c[i];
        }
    }
};

// Specialization for tensor core operations
template<>
struct MmaOperator<
    cutlass::half_t,    // ElementA
    cutlass::half_t,    // ElementB
    float,              // ElementC
    cutlass::arch::Sm80 // ArchTag
> {
    CUTLASS_DEVICE
    void operator()(
        Array<float, 8> &d,
        Array<cutlass::half_t, 8> const &a,
        Array<cutlass::half_t, 8> const &b,
        Array<float, 8> const &c
    ) {
        // Use tensor core instruction
        int const kGroups = 2;
        
        CUTLASS_PRAGMA_UNROLL
        for (int g = 0; g < kGroups; ++g) {
            // Pack arguments for tensor core instruction
            uint32_t const *A = reinterpret_cast<uint32_t const *>(&a) + g * 2;
            uint32_t const *B = reinterpret_cast<uint32_t const *>(&b) + g * 2;
            float const *C = reinterpret_cast<float const *>(&c) + g * 4;
            float *D = reinterpret_cast<float *>(&d) + g * 4;
            
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0, %1, %2, %3}, {%4, %5}, {%6, %7}, {%8, %9, %10, %11};"
                : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])
                : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]),
                  "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3])
            );
        }
    }
};
```

## Topic 6: Memory Access Pattern Optimization

CUTLASS optimizes memory access patterns through template-based approaches.

### Memory Access Traits
```cpp
// Memory access pattern traits
template<typename Element, typename Layout>
struct MemoryAccessTraits {
    static int const kAccessSize = sizeof(Element);
    static int const kElementsPerAccess = 1;
    static bool const kRequiresContiguousLoad = true;
};

// Specialization for vectorized loads
template<>
struct MemoryAccessTraits<float, cutlass::layout::ColumnMajor> {
    static int const kAccessSize = sizeof(float) * 4;  // 4 floats at once
    static int const kElementsPerAccess = 4;
    static bool const kRequiresContiguousLoad = true;
    
    CUTLASS_HOST_DEVICE
    static Array<float, 4> load(void const *ptr) {
        Array<float, 4> result;
        // Use vectorized load instruction
        reinterpret_cast<Array<float, 4> const*>(ptr)[0] = result;
        return result;
    }
};

// Coalescing helper
template<int kAccessSize, int kWarpSize = 32>
struct CoalescingTraits {
    static int const kElementsPerWarp = (kWarpSize * kAccessSize) / sizeof(int);
    static bool const kIsCoalesced = (kAccessSize % 4) == 0;  // Word-aligned
};
```

### Shared Memory Banking
```cpp
// Shared memory banking optimization
template<int Rows, int Cols, typename Element>
struct SharedMemoryMatrix {
    Element storage[Rows][Cols];
    
    // Access function that considers banking
    CUTLASS_DEVICE
    Element& access(int row, int col) {
        // Add padding to avoid bank conflicts
        static constexpr int kPaddedCols = Cols + (Cols % 32 == 0 ? 1 : 0);
        return reinterpret_cast<Element(&)[Rows][kPaddedCols]>(storage)[row][col];
    }
};

// Specialization for common cases
template<typename Element>
struct SharedMemoryMatrix<16, 16, Element> {
    // 16x16 is a common tensor core fragment size
    Element storage[16][17];  // +1 to avoid bank conflicts
    
    CUTLASS_DEVICE
    Element& access(int row, int col) {
        return storage[row][col];
    }
};
```

## Hands-on Exercises

### Exercise 1: Custom GEMM Template
Create a simplified GEMM template following CUTLASS conventions.

```cpp
// TODO: Create a GEMM template with the following specifications:
// 1. Template parameters for ElementA, LayoutA, ElementB, LayoutB, ElementC, LayoutC
// 2. Include operator class and architecture tags
// 3. Add threadblock shape parameter
// 4. Implement basic GEMM operation with proper template structure
// 5. Include appropriate type aliases and constants
```

### Exercise 2: Architecture Dispatch System
Implement a dispatch system similar to CUTLASS for selecting implementations.

```cpp
// TODO: Create a dispatch system that:
// 1. Uses tag-based dispatch for different architectures
// 2. Selects different implementations based on hardware features
// 3. Includes specializations for common cases
// 4. Demonstrates conditional compilation based on architecture
```

### Exercise 3: Memory Access Optimization
Create optimized memory access patterns for different data types.

```cpp
// TODO: Implement memory access traits that:
// 1. Optimize for different element types (half, float, int8)
// 2. Consider coalescing patterns
// 3. Include vectorized access options
// 4. Handle different memory layouts efficiently
```

## Solutions to Exercises

### Solution 1: Custom GEMM Template
```cpp
#include <cutlass/array.h>
#include <cutlass/layout/matrix.h>

// Custom GEMM template following CUTLASS conventions
template<
    typename ElementA_,
    typename LayoutA_,
    typename ElementB_,
    typename LayoutB_,
    typename ElementC_,
    typename LayoutC_,
    typename ElementAccumulator_ = ElementC_,
    typename OperatorClass_ = cutlass::arch::OpClassSimt,
    typename ArchTag_ = cutlass::arch::Sm70,
    typename ThreadblockShape_ = cutlass::gemm::GemmShape<128, 128, 32>,
    int AlignmentA_ = 128 / cutlass::sizeof_bits<ElementA_>::value,
    int AlignmentB_ = 128 / cutlass::sizeof_bits<ElementB_>::value,
    int AlignmentC_ = 128 / cutlass::sizeof_bits<ElementC_>::value
>
class CustomGemmTemplate {
public:
    using ElementA = ElementA_;
    using LayoutA = LayoutA_;
    using ElementB = ElementB_;
    using LayoutB = LayoutB_;
    using ElementC = ElementC_;
    using LayoutC = LayoutC_;
    using ElementAccumulator = ElementAccumulator_;
    using OperatorClass = OperatorClass_;
    using ArchTag = ArchTag_;
    using ThreadblockShape = ThreadblockShape_;
    
    static int const kAlignmentA = AlignmentA_;
    static int const kAlignmentB = AlignmentB_;
    static int const kAlignmentC = AlignmentC_;
    
    // Derived types
    using LayoutAIndirect = typename cutlass::layout::PitchLinear;
    using LayoutBIndirect = typename cutlass::layout::PitchLinear;
    
    // Constants
    static int const kM = ThreadblockShape::kM;
    static int const kN = ThreadblockShape::kN;
    static int const kK = ThreadblockShape::kK;
    
    // Threadblock-level tile count
    static int const kThreadblockCountM = 1;
    static int const kThreadblockCountN = 1;
    
    // Verification
    static bool const kIsSimt = cutlass::arch::OpClassSimt == OperatorClass::kTag;
    static bool const kIsTensorOp = cutlass::arch::OpClassTensorOp == OperatorClass::kTag;
    
    // Main GEMM operation
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        ElementA const *ptr_A;
        ElementB const *ptr_B;
        ElementC const *ptr_C;
        ElementC *ptr_D;
        typename LayoutA::Stride::Index lda;
        typename LayoutB::Stride::Index ldb;
        typename LayoutC::Stride::Index ldc;
        typename LayoutC::Stride::Index ldd;
        ElementC alpha;
        ElementC beta;
    };
    
    // Kernel entry point would be defined here
    // using CUTLASS_DEVICE functions and proper threadblock coordination
};
```

### Solution 2: Architecture Dispatch System
```cpp
// Architecture tags
struct Compute_70 {};
struct Compute_75 {};
struct Compute_80 {};
struct Compute_90 {};

// Feature flags for each architecture
template<typename ArchTag>
struct ArchFeatures;

template<>
struct ArchFeatures<Compute_80> {
    static constexpr bool has_tensor_cores = true;
    static constexpr bool has_sparse_tensor_cores = true;
    static constexpr int max_shared_memory = 164 * 1024;
    static constexpr int max_threads_per_sm = 2048;
    static constexpr int registers_per_thread = 255;
};

template<>
struct ArchFeatures<Compute_75> {
    static constexpr bool has_tensor_cores = true;
    static constexpr bool has_sparse_tensor_cores = false;
    static constexpr int max_shared_memory = 64 * 1024;
    static constexpr int max_threads_per_sm = 1024;
    static constexpr int registers_per_thread = 65536;
};

template<>
struct ArchFeatures<Compute_70> {
    static constexpr bool has_tensor_cores = false;
    static constexpr bool has_sparse_tensor_cores = false;
    static constexpr int max_shared_memory = 96 * 1024;
    static constexpr int max_threads_per_sm = 512;
    static constexpr int registers_per_thread = 65536;
};

// Dispatcher for different implementations
template<typename ArchTag>
struct KernelDispatcher {
    template<typename Element>
    static CUTLASS_DEVICE void execute_kernel(Element* data, int size) {
        if constexpr (ArchFeatures<ArchTag>::has_tensor_cores) {
            // Use tensor core optimized version
            tensor_core_kernel<Element>(data, size);
        } else {
            // Use SIMT optimized version
            simt_kernel<Element>(data, size);
        }
    }
    
private:
    template<typename Element>
    static CUTLASS_DEVICE void tensor_core_kernel(Element* data, int size) {
        // Tensor core optimized implementation
        for (int i = 0; i < size; i += 16) {  // Tensor core operates on 16x16 tiles
            // Perform tensor core operations
        }
    }
    
    template<typename Element>
    static CUTLASS_DEVICE void simt_kernel(Element* data, int size) {
        // SIMT optimized implementation
        for (int i = 0; i < size; ++i) {
            // Perform regular operations
        }
    }
};

// Usage example
template<typename ArchTag>
CUTLASS_GLOBAL void kernel_launcher(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx == 0) {
        KernelDispatcher<ArchTag>::execute_kernel(data, size);
    }
}
```

### Solution 3: Memory Access Optimization
```cpp
#include <cutlass/array.h>

// Memory access traits for optimization
template<typename Element, typename Layout>
struct OptimizedMemoryAccess {
    using AccessType = Element;
    static int const kElementsPerAccess = 1;
    static int const kAccessSize = sizeof(Element);
    
    CUTLASS_DEVICE
    static Element load(Element const *ptr, int idx) {
        return ptr[idx];
    }
    
    CUTLASS_DEVICE
    static void store(Element *ptr, int idx, Element value) {
        ptr[idx] = value;
    }
};

// Specialization for vectorized access with float4
template<>
struct OptimizedMemoryAccess<float, cutlass::layout::ColumnMajor> {
    using AccessType = cutlass::Array<float, 4>;
    static int const kElementsPerAccess = 4;
    static int const kAccessSize = sizeof(float) * 4;
    
    CUTLASS_DEVICE
    static cutlass::Array<float, 4> load(float const *ptr, int base_idx) {
        cutlass::Array<float, 4> result;
        float4 temp = reinterpret_cast<float4 const*>(ptr)[base_idx];
        result[0] = temp.x;
        result[1] = temp.y;
        result[2] = temp.z;
        result[3] = temp.w;
        return result;
    }
    
    CUTLASS_DEVICE
    static void store(float *ptr, int base_idx, cutlass::Array<float, 4> const &value) {
        float4 temp;
        temp.x = value[0];
        temp.y = value[1];
        temp.z = value[2];
        temp.w = value[3];
        reinterpret_cast<float4*>(ptr)[base_idx] = temp;
    }
};

// Specialization for half precision with vectorized access
template<>
struct OptimizedMemoryAccess<cutlass::half_t, cutlass::layout::ColumnMajor> {
    using AccessType = cutlass::Array<cutlass::half_t, 8>;  // 8 halfs = 16 bytes = 128 bits
    static int const kElementsPerAccess = 8;
    static int const kAccessSize = sizeof(cutlass::half_t) * 8;
    
    CUTLASS_DEVICE
    static cutlass::Array<cutlass::half_t, 8> load(cutlass::half_t const *ptr, int base_idx) {
        cutlass::Array<cutlass::half_t, 8> result;
        uint4 temp = reinterpret_cast<uint4 const*>(ptr)[base_idx];
        // Pack 8 half_ts into uint4 for vectorized load
        reinterpret_cast<uint4*>(result.data())[0] = temp;
        return result;
    }
    
    CUTLASS_DEVICE
    static void store(cutlass::half_t *ptr, int base_idx, cutlass::Array<cutlass::half_t, 8> const &value) {
        uint4 temp = reinterpret_cast<uint4 const*>(value.data())[0];
        reinterpret_cast<uint4*>(ptr)[base_idx] = temp;
    }
};

// Coalescing helper for different access patterns
template<typename Element, typename Layout>
struct CoalescingHelper {
    CUTLASS_DEVICE
    static int get_coalesced_address(int thread_id, int element_idx, int stride) {
        // For column-major: consecutive threads access consecutive rows
        // This gives good coalescing when accessing the same column
        return thread_id * stride + element_idx;
    }
};

template<>
struct CoalescingHelper<float, cutlass::layout::RowMajor> {
    CUTLASS_DEVICE
    static int get_coalesced_address(int thread_id, int element_idx, int stride) {
        // For row-major: consecutive threads access consecutive columns
        // This gives good coalescing when accessing the same row
        return element_idx * stride + thread_id;
    }
};
```

## Advanced Topic: CUTLASS Template Metaprogramming Techniques

### Conditional Compilation with Templates
```cpp
// CUTLASS-style conditional compilation
template<bool kCondition, typename T = void>
struct EnableIfC {
    using type = T;
};

template<typename T>
struct EnableIfC<false, T> {};

template<bool Condition, typename T = void>
using EnableIfC_t = typename EnableIfC<Condition, T>::type;

// Usage in CUTLASS context
template<typename ElementA, typename ElementB, typename ElementC, typename ArchTag>
struct ConditionalGemmImplementation {
    template<typename = void>
    CUTLASS_DEVICE 
    static EnableIfC_t<ArchTraits<ArchTag>::kSupportsTensorOps> 
    execute_with_tensor_cores(ElementC& d, ElementA a, ElementB b, ElementC c) {
        // Tensor core implementation
        tensor_core_mma(d, a, b, c);
    }
    
    template<typename = void>
    CUTLASS_DEVICE 
    static EnableIfC_t<!ArchTraits<ArchTag>::kSupportsTensorOps> 
    execute_with_tensor_cores(ElementC& d, ElementA a, ElementB b, ElementC c) {
        // Regular math implementation
        d = a * b + c;
    }
    
    CUTLASS_DEVICE
    static void execute(ElementC& d, ElementA a, ElementB b, ElementC c) {
        execute_with_tensor_cores(d, a, b, c);
    }
};
```

## Quiz Questions

1. What is the typical order of template parameters in CUTLASS?

2. How does CUTLASS use tag-based dispatch for different architectures?

3. What is the purpose of template specialization in CUTLASS?

4. How does CUTLASS handle hardware-specific optimizations?

5. What role do memory access patterns play in CUTLASS performance?

## Summary
Module 7 explored CUTLASS-specific template patterns and idioms, including template parameter conventions, dispatch patterns, specialization strategies, hardware optimizations, math instruction integration, and memory access optimizations. These patterns are essential for understanding and extending CUTLASS effectively.