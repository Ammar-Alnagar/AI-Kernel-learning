# Module 8: Advanced CUTLASS Customization

## Overview
This module covers advanced techniques for extending and customizing CUTLASS for specific use cases, including custom epilogues, non-standard data types, tensor operations beyond GEMM, and performance tuning strategies.

## Learning Objectives
By the end of this module, students will be able to:
- Implement custom epilogue operations for specialized computations
- Add support for non-standard data types in CUTLASS
- Extend CUTLASS beyond GEMM to other tensor operations
- Apply performance tuning strategies for specific hardware
- Debug and profile template-heavy CUTLASS code
- Integrate CUTLASS with other CUDA libraries

## Topic 1: Custom Epilogue Operations

Custom epilogues allow fusing additional operations with the main GEMM computation, improving performance by reducing memory traffic.

### Basic Custom Epilogue
```cpp
#include <cutlass/epilogue/thread/linear_combination.h>
#include <cutlass/array.h>

// Custom epilogue: D = alpha * A * B + beta * C + gamma * bias
template<typename ElementOutput, typename ElementAccumulator>
class LinearCombinationWithBias {
private:
    ElementOutput alpha_;
    ElementOutput beta_;
    ElementOutput gamma_;
    ElementOutput const *bias_ptr_;
    int stride_bias_;

public:
    using ElementCompute = ElementOutput;
    using ElementAccumulatorInOut = ElementAccumulator;
    
    struct Arguments {
        ElementCompute alpha;
        ElementCompute beta;
        ElementCompute gamma;
        ElementOutput const *bias = nullptr;
        int stride_bias = 0;
    };

    CUTLASS_HOST
    Status initialize(Arguments const &args) {
        alpha_ = args.alpha;
        beta_ = args.beta;
        gamma_ = args.gamma;
        bias_ptr_ = args.bias;
        stride_bias_ = args.stride_bias;
        return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    LinearCombinationWithBias(Arguments const &args) {
        initialize(args);
    }

    CUTLASS_HOST_DEVICE
    LinearCombinationWithBias(
        ElementCompute alpha,
        ElementCompute beta,
        ElementCompute gamma = ElementCompute(1),
        ElementOutput const *bias = nullptr,
        int stride_bias = 0) :
        alpha_(alpha), beta_(beta), gamma_(gamma), bias_ptr_(bias), stride_bias_(stride_bias) {}

    CUTLASS_HOST_DEVICE
    LinearCombinationWithBias() : alpha_(1), beta_(0), gamma_(1), bias_ptr_(nullptr), stride_bias_(0) {}

    /// Functionally identical to operator()
    CUTLASS_HOST_DEVICE
    Array<ElementOutput, 1> operator()(Array<ElementAccumulator, 1> const &frag_C,
                                      Array<ElementAccumulator, 1> const &frag_accum) const {
        Array<ElementOutput, 1> frag_D;
        
        ElementOutput intermediate = alpha_ * ElementOutput(frag_accum[0]) + 
                                   beta_ * ElementOutput(frag_C[0]);
        
        if (bias_ptr_) {
            intermediate += gamma_ * bias_ptr_[0];
        }
        
        frag_D[0] = intermediate;
        return frag_D;
    }

    /// Special handling for implicit broadcasting with 1x1 tile
    CUTLASS_HOST_DEVICE
    Array<ElementOutput, 1> operator()(Array<ElementAccumulator, 1> const &frag_accum) const {
        Array<ElementOutput, 1> frag_D;
        
        ElementOutput intermediate = alpha_ * ElementOutput(frag_accum[0]);
        
        if (bias_ptr_) {
            intermediate += gamma_ * bias_ptr_[0];
        }
        
        frag_D[0] = intermediate;
        return frag_D;
    }
};
```

### Advanced Epilogue with Activation Functions
```cpp
// Epilogue with activation function
template<typename ElementOutput, typename ElementAccumulator, typename ActivationFunctor>
class LinearCombinationWithActivation {
private:
    ElementOutput alpha_;
    ElementOutput beta_;
    ActivationFunctor activation_;

public:
    using ElementCompute = ElementOutput;
    using ElementAccumulatorInOut = ElementAccumulator;
    
    struct Arguments {
        ElementCompute alpha;
        ElementCompute beta;
        ActivationFunctor activation;
    };

    CUTLASS_HOST
    Status initialize(Arguments const &args) {
        alpha_ = args.alpha;
        beta_ = args.beta;
        activation_ = args.activation;
        return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    LinearCombinationWithActivation(Arguments const &args) {
        initialize(args);
    }

    CUTLASS_HOST_DEVICE
    LinearCombinationWithActivation(
        ElementCompute alpha,
        ElementCompute beta,
        ActivationFunctor activation = ActivationFunctor()) :
        alpha_(alpha), beta_(beta), activation_(activation) {}

    CUTLASS_HOST_DEVICE
    Array<ElementOutput, 1> operator()(Array<ElementAccumulator, 1> const &frag_C,
                                      Array<ElementAccumulator, 1> const &frag_accum) const {
        Array<ElementOutput, 1> frag_D;
        
        ElementOutput intermediate = alpha_ * ElementOutput(frag_accum[0]) + 
                                   beta_ * ElementOutput(frag_C[0]);
        
        frag_D[0] = activation_(intermediate);
        return frag_D;
    }
};

// Example activation functors
struct Sigmoid {
    template<typename T>
    CUTLASS_HOST_DEVICE T operator()(T x) const {
        return T(1) / (T(1) + exp(-x));
    }
};

struct Tanh {
    template<typename T>
    CUTLASS_HOST_DEVICE T operator()(T x) const {
        T exp_x = exp(x);
        T exp_neg_x = exp(-x);
        return (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
    }
};

struct Gelu {
    template<typename T>
    CUTLASS_HOST_DEVICE T operator()(T x) const {
        return T(0.5) * x * (T(1) + tanh(sqrt(T(2) / T(M_PI)) * (x + T(0.044715) * x * x * x)));
    }
};
```

## Topic 2: Non-Standard Data Types Support

Extending CUTLASS to support custom data types requires implementing proper converters and numeric operations.

### Custom Data Type Definition
```cpp
// Example: Custom 4-bit quantized type
struct int4b_t {
    int4b_t() = default;
    
    CUTLASS_HOST_DEVICE
    int4b_t(int8_t value) : data_(static_cast<int8_t>(value << 4) >> 4) {}  // Sign extend
    
    CUTLASS_HOST_DEVICE
    operator int8_t() const { return data_; }
    
    int8_t data_;
};

// Converter for int4b_t
template<>
struct NumericConverter<int4b_t, float> {
    using round_style = cutlass::FloatRoundStyle::round_to_nearest;
    
    CUTLASS_HOST_DEVICE
    static int4b_t convert(float src) {
        int8_t rounded = static_cast<int8_t>(roundf(src));
        rounded = max(-8, min(7, rounded));  // Clamp to 4-bit range
        return int4b_t(rounded);
    }
    
    CUTLASS_HOST_DEVICE
    int4b_t operator()(float src) {
        return convert(src);
    }
};

template<>
struct NumericConverter<float, int4b_t> {
    CUTLASS_HOST_DEVICE
    static float convert(int4b_t src) {
        return static_cast<float>(src.data_);
    }
    
    CUTLASS_HOST_DEVICE
    float operator()(int4b_t src) {
        return convert(src);
    }
};
```

### Custom Numeric Operations
```cpp
// Operations for custom types
template<typename T>
struct NumericOperations;

template<>
struct NumericOperations<int4b_t> {
    CUTLASS_HOST_DEVICE
    static int4b_t multiply(int4b_t a, int4b_t b) {
        int8_t result = static_cast<int8_t>(a.data_) * static_cast<int8_t>(b.data_);
        return int4b_t(result);
    }
    
    CUTLASS_HOST_DEVICE
    static int4b_t add(int4b_t a, int4b_t b) {
        int8_t result = static_cast<int8_t>(a.data_) + static_cast<int8_t>(b.data_);
        return int4b_t(result);
    }
    
    CUTLASS_HOST_DEVICE
    static int4b_t multiply_add(int4b_t a, int4b_t b, int4b_t c) {
        return add(multiply(a, b), c);
    }
};
```

## Topic 3: Tensor Operations Beyond GEMM

CUTLASS can be extended to support operations beyond basic GEMM, such as convolution, batched operations, and tensor contractions.

### Batched GEMM Extension
```cpp
#include <cutlass/gemm/device/gemm_batched.h>

// Custom batched GEMM with different strides
template<typename ElementA, typename ElementB, typename ElementC>
class CustomBatchedGemm {
public:
    using GemmType = cutlass::gemm::device::GemmBatched<
        ElementA, cutlass::layout::ColumnMajor,
        ElementB, cutlass::layout::ColumnMajor,
        ElementC, cutlass::layout::ColumnMajor,
        ElementC
    >;
    
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        ElementA const *ptr_A;
        ElementB const *ptr_B;
        ElementC const *ptr_C;
        ElementC *ptr_D;
        int64_t batch_stride_A;
        int64_t batch_stride_B;
        int64_t batch_stride_C;
        int64_t batch_stride_D;
        typename GemmType::EpilogueOutputOp::Params epilogue;
        int batch_count;
        int split_k_slices;
    };
    
    GemmType gemm_operator;
    
    Status operator()(Arguments const &args) {
        typename GemmType::Arguments device_args{
            args.problem_size,
            {args.ptr_A, args.problem_size.m(), args.problem_size.k()},
            {args.ptr_B, args.problem_size.k(), args.problem_size.n()},
            {args.ptr_C, args.problem_size.m(), args.problem_size.n()},
            {args.ptr_D, args.problem_size.m(), args.problem_size.n()},
            args.epilogue,
            args.batch_count,
            args.batch_stride_A,
            args.batch_stride_B,
            args.batch_stride_C,
            args.batch_stride_D,
            args.split_k_slices
        };
        
        return gemm_operator(device_args);
    }
};
```

### Convolution as GEMM
```cpp
// Convert convolution to GEMM using im2col transformation
template<typename ElementInput, typename ElementWeight, typename ElementOutput>
class ConvAsGemm {
private:
    // Temporary storage for im2col transformed data
    ElementInput *transformed_input_;
    size_t transformed_size_;
    
public:
    // GEMM operator for the transformed data
    using GemmOperator = cutlass::gemm::device::Gemm<
        ElementWeight, cutlass::layout::ColumnMajor,
        ElementInput, cutlass::layout::ColumnMajor,
        ElementOutput, cutlass::layout::ColumnMajor,
        ElementOutput
    >;
    
    GemmOperator gemm_op;
    
    Status run_conv_as_gemm(
        int batch_size, int channels, int height, int width,
        int kernel_h, int kernel_w,
        ElementInput const *input,
        ElementWeight const *weight,
        ElementOutput *output) {
        
        // Calculate transformed dimensions
        int output_h = height - kernel_h + 1;
        int output_w = width - kernel_w + 1;
        int transformed_k = channels * kernel_h * kernel_w;
        int transformed_n = batch_size * output_h * output_w;
        
        // Allocate temporary storage for transformed input
        size_t temp_bytes = transformed_k * transformed_n * sizeof(ElementInput);
        cudaMalloc(&transformed_input_, temp_bytes);
        
        // Perform im2col transformation
        im2col_transform(input, transformed_input_, 
                        batch_size, channels, height, width,
                        kernel_h, kernel_w, output_h, output_w);
        
        // Run GEMM: output = weight * transformed_input
        cutlass::gemm::GemmCoord problem_size(transformed_k, transformed_n, channels);
        
        typename GemmOperator::Arguments arguments{
            problem_size,
            {weight, transformed_k, channels},  // Weight matrix (K x C)
            {transformed_input_, transformed_k, transformed_n},  // Input matrix (K x N)
            {output, channels, transformed_n},  // Output matrix (C x N)
            {output, channels, transformed_n},  // Output matrix (C x N)
            {1.0f, 0.0f}  // Alpha, Beta
        };
        
        Status status = gemm_op(arguments);
        
        // Cleanup
        cudaFree(transformed_input_);
        
        return status;
    }
    
private:
    void im2col_transform(
        ElementInput const *input,
        ElementInput *output,
        int batch_size, int channels, int height, int width,
        int kernel_h, int kernel_w,
        int output_h, int output_w) {
        
        // Perform the im2col transformation
        // This is a simplified version - in practice, this would be a kernel
        for (int b = 0; b < batch_size; ++b) {
            for (int c = 0; c < channels; ++c) {
                for (int kh = 0; kh < kernel_h; ++kh) {
                    for (int kw = 0; kw < kernel_w; ++kw) {
                        for (int oh = 0; oh < output_h; ++oh) {
                            for (int ow = 0; ow < output_w; ++ow) {
                                int input_h = oh + kh;
                                int input_w = ow + kw;
                                
                                int input_idx = ((b * channels + c) * height + input_h) * width + input_w;
                                int output_idx = ((c * kernel_h + kh) * kernel_w + kw) * (batch_size * output_h * output_w) + 
                                                ((b * output_h + oh) * output_w + ow);
                                
                                output[output_idx] = input[input_idx];
                            }
                        }
                    }
                }
            }
        }
    }
};
```

## Topic 4: Performance Tuning Strategies

Performance tuning in CUTLASS involves optimizing various aspects of the computation.

### Kernel Configuration Tuning
```cpp
// Performance tuner for different problem sizes
template<typename ElementA, typename ElementB, typename ElementC>
class PerformanceTuner {
public:
    struct Config {
        cutlass::gemm::GemmCoord threadblock_shape;
        cutlass::gemm::GemmCoord warp_shape;
        cutlass::gemm::GemmShape instruction_shape;
        int stages;
        bool use_split_k;
    };
    
    static Config get_optimal_config(
        cutlass::gemm::GemmCoord problem_size,
        cutlass::arch::Architecture arch) {
        
        Config config;
        
        // Heuristic-based configuration selection
        if (problem_size.m() >= 2048 && problem_size.n() >= 2048) {
            // Large problem: use larger tiles
            config.threadblock_shape = cutlass::gemm::GemmShape<256, 128, 32>;
            config.warp_shape = cutlass::gemm::GemmShape<64, 64, 32>;
            config.stages = 4;
        } else if (problem_size.m() <= 512 && problem_size.n() <= 512) {
            // Small problem: use smaller tiles to increase occupancy
            config.threadblock_shape = cutlass::gemm::GemmShape<128, 128, 32>;
            config.warp_shape = cutlass::gemm::GemmShape<64, 64, 32>;
            config.stages = 3;
        } else {
            // Medium problem: balanced approach
            config.threadblock_shape = cutlass::gemm::GemmShape<128, 256, 32>;
            config.warp_shape = cutlass::gemm::GemmShape<64, 64, 32>;
            config.stages = 3;
        }
        
        // Set instruction shape based on architecture
        if (arch >= cutlass::arch::Sm80) {
            config.instruction_shape = cutlass::gemm::GemmShape<16, 8, 16>;  // For FP16
        } else {
            config.instruction_shape = cutlass::gemm::GemmShape<16, 16, 16>; // For older arch
        }
        
        // Determine if split-K is beneficial
        config.use_split_k = (problem_size.k() > 8192);
        
        return config;
    }
    
    // Benchmark different configurations
    template<typename GemmOperator>
    static float benchmark_config(
        typename GemmOperator::Arguments const &args,
        Config const &config) {
        
        // Setup timer
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warm up
        GemmOperator op;
        op(args);
        cudaDeviceSynchronize();
        
        // Time the operation
        cudaEventRecord(start);
        for (int i = 0; i < 10; ++i) {  // Run multiple times for average
            op(args);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds / 10.0f;  // Average time
    }
};
```

### Memory Bandwidth Optimization
```cpp
// Memory access pattern optimizer
template<typename Element, typename Layout>
class MemoryOptimizer {
public:
    // Check if the memory access pattern is optimal
    static bool is_coalesced_access(int access_stride, int element_size) {
        // For coalesced access, consecutive threads should access consecutive memory
        int bytes_per_access = access_stride * element_size;
        return (bytes_per_access % 128) == 0;  // Should be 128-byte aligned for optimal performance
    }
    
    // Calculate optimal padding to avoid bank conflicts
    template<int ElementsPerRow>
    static constexpr int get_optimal_padding() {
        // Add padding to avoid shared memory bank conflicts
        // Typically add 1 element per 32 elements to avoid conflicts
        return (ElementsPerRow % 32 == 0) ? 1 : 0;
    }
    
    // Reorder data for better cache locality
    template<int BlockSize>
    static void reorder_for_cache_locality(Element* data, int rows, int cols) {
        // Block-wise reordering to improve cache performance
        for (int i = 0; i < rows; i += BlockSize) {
            for (int j = 0; j < cols; j += BlockSize) {
                // Process block [i:i+BlockSize, j:j+BlockSize]
                for (int bi = i; bi < min(i + BlockSize, rows); ++bi) {
                    for (int bj = j; bj < min(j + BlockSize, cols); ++bj) {
                        // Access data[bi][bj] in block order
                        Element temp = data[bi * cols + bj];
                        // Process temp...
                    }
                }
            }
        }
    }
};
```

## Topic 5: Debugging Template-Heavy Code

Debugging complex template metaprogramming in CUTLASS requires special techniques.

### Template Debugging Utilities
```cpp
// Utility for debugging template instantiations
template<typename T>
struct TypePrinter;

// GCC/Clang trick to force compiler to show type
template<typename T>
void debug_type() {
    // This will cause a compilation error that shows the type T
    // TypePrinter<T> printer;  // Uncomment to see type in error message
}

// Static assertions for debugging
template<typename T>
struct DebugInfo {
    static_assert(std::is_arithmetic_v<T>, "T must be arithmetic");
    static_assert(sizeof(T) >= 4, "T must be at least 4 bytes");
    
    // Print size at compile time
    static constexpr size_t size = sizeof(T);
};

// Runtime debugging helpers
template<typename Element>
class DebugGemm {
public:
    static void validate_arguments(
        Element const *A, Element const *B, Element const *C, Element *D,
        int M, int N, int K,
        int lda, int ldb, int ldc, int ldd) {
        
        // Validate pointers are not null
        assert(A != nullptr && B != nullptr && C != nullptr && D != nullptr);
        
        // Validate leading dimensions
        assert(lda >= M && ldb >= K && ldc >= M && ldd >= M);
        
        // Validate problem size
        assert(M > 0 && N > 0 && K > 0);
        
        // Validate memory alignment if required
        if constexpr (alignof(Element) >= 16) {
            assert(reinterpret_cast<uintptr_t>(A) % 16 == 0);
            assert(reinterpret_cast<uintptr_t>(B) % 16 == 0);
            assert(reinterpret_cast<uintptr_t>(C) % 16 == 0);
            assert(reinterpret_cast<uintptr_t>(D) % 16 == 0);
        }
    }
    
    // Print problem information
    static void print_problem_info(int M, int N, int K) {
        printf("GEMM Problem: %dx%d = %dx%d * %dx%d\n", M, N, M, K, K, N);
        printf("Total operations: %ld\n", static_cast<long>(M) * N * K * 2);
        printf("Memory footprint: %ld bytes\n", 
               static_cast<long>(M * K + K * N + M * N) * sizeof(Element));
    }
};
```

### Profiling and Performance Analysis
```cpp
#include <nvtx3/nvToolsExt.h>

// Profiling utilities for CUTLASS operations
class CutlassProfiler {
public:
    struct ProfileData {
        float elapsed_ms;
        size_t bytes_transferred;
        size_t flops_performed;
        float bandwidth_gbs;
        float gflops;
    };
    
    static ProfileData profile_gemm_operation(
        std::function<void()> const &operation,
        size_t bytes_read,
        size_t bytes_written,
        size_t flops) {
        
        ProfileData data;
        
        // Start NVTX range for profiling
        nvtxRangePushA("CUTLASS_GEMM");
        
        // Setup timing
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        // Warm up
        operation();
        cudaDeviceSynchronize();
        
        // Time the operation
        cudaEventRecord(start);
        operation();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&data.elapsed_ms, start, stop);
        
        // Calculate metrics
        data.bytes_transferred = bytes_read + bytes_written;
        data.flops_performed = flops;
        data.bandwidth_gbs = (data.bytes_transferred / 1e9) / (data.elapsed_ms / 1000.0f);
        data.gflops = (data.flops_performed / 1e9) / (data.elapsed_ms / 1000.0f);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        nvtxRangePop();
        
        return data;
    }
    
    // Memory access pattern analysis
    template<typename Element>
    static void analyze_memory_access_pattern(
        Element const *ptr, int count, const char* name) {
        
        // Log memory access information
        printf("Memory analysis for %s:\n", name);
        printf("  Address: %p\n", static_cast<const void*>(ptr));
        printf("  Size: %zu bytes\n", count * sizeof(Element));
        printf("  Alignment: %zu bytes\n", 
               reinterpret_cast<uintptr_t>(ptr) % alignof(Element));
    }
};
```

## Hands-on Exercises

### Exercise 1: Custom Epilogue with Softmax
Implement a custom epilogue that applies softmax normalization to the output.

```cpp
// TODO: Create a custom epilogue that implements:
// 1. Row-wise softmax normalization: exp(x_i) / sum(exp(x_j)) for all j in row
// 2. Proper numerical stability (subtract max before exp)
// 3. Support for different data types
// 4. Integration with CUTLASS epilogue interface
```

### Exercise 2: Quantized GEMM
Extend CUTLASS to support INT8 quantized GEMM operations.

```cpp
// TODO: Implement INT8 quantized GEMM with:
// 1. Custom data type for INT8 with proper converters
// 2. Quantization/dequantization in epilogue
// 3. Proper scaling factors
// 4. Integration with existing CUTLASS infrastructure
```

### Exercise 3: Performance Tuning Framework
Create a framework for automatically tuning CUTLASS parameters.

```cpp
// TODO: Create a performance tuning framework that:
// 1. Tests multiple configurations for a given problem size
// 2. Benchmarks each configuration
// 3. Selects the optimal one based on performance
// 4. Caches results for future use
```

## Solutions to Exercises

### Solution 1: Custom Epilogue with Softmax
```cpp
#include <cutlass/numeric_conversion.h>

// Custom epilogue with softmax normalization
template<typename ElementOutput, typename ElementAccumulator>
class LinearCombinationSoftmax {
private:
    ElementOutput alpha_;
    ElementOutput beta_;
    
public:
    using ElementCompute = ElementOutput;
    using ElementAccumulatorInOut = ElementAccumulator;
    
    struct Arguments {
        ElementCompute alpha;
        ElementCompute beta;
    };

    CUTLASS_HOST
    Status initialize(Arguments const &args) {
        alpha_ = args.alpha;
        beta_ = args.beta;
        return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    LinearCombinationSoftmax(Arguments const &args) {
        initialize(args);
    }

    CUTLASS_HOST_DEVICE
    LinearCombinationSoftmax(
        ElementCompute alpha = ElementCompute(1),
        ElementCompute beta = ElementCompute(0)) :
        alpha_(alpha), beta_(beta) {}

    CUTLASS_HOST_DEVICE
    Array<ElementOutput, 1> operator()(Array<ElementAccumulator, 1> const &frag_C,
                                      Array<ElementAccumulator, 1> const &frag_accum) const {
        Array<ElementOutput, 1> frag_D;
        
        // For softmax, we need to consider the entire row
        // This is a simplified version for single-element processing
        ElementOutput intermediate = alpha_ * ElementOutput(frag_accum[0]) + 
                                   beta_ * ElementOutput(frag_C[0]);
        
        // In a real implementation, softmax would need to process entire rows
        // This is a placeholder for the actual softmax computation
        frag_D[0] = intermediate;
        return frag_D;
    }
};

// More complete softmax implementation for vectors
template<typename ElementOutput, typename ElementAccumulator>
class VectorSoftmaxEpilogue {
private:
    ElementOutput alpha_;
    ElementOutput beta_;
    
public:
    using ElementCompute = ElementOutput;
    using ElementAccumulatorInOut = ElementAccumulator;
    
    struct Arguments {
        ElementCompute alpha;
        ElementCompute beta;
    };

    CUTLASS_HOST
    Status initialize(Arguments const &args) {
        alpha_ = args.alpha;
        beta_ = args.beta;
        return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    VectorSoftmaxEpilogue(Arguments const &args) {
        initialize(args);
    }

    // This would be called for each row of the output matrix
    CUTLASS_DEVICE
    void apply_softmax(ElementOutput* row_data, int row_length) const {
        // Find maximum value for numerical stability
        ElementOutput max_val = row_data[0];
        for (int i = 1; i < row_length; ++i) {
            if (row_data[i] > max_val) {
                max_val = row_data[i];
            }
        }
        
        // Compute exp(x - max) and sum
        ElementOutput sum = ElementOutput(0);
        for (int i = 0; i < row_length; ++i) {
            ElementOutput exp_val = exp(row_data[i] - max_val);
            row_data[i] = exp_val;
            sum += exp_val;
        }
        
        // Normalize by sum
        for (int i = 0; i < row_length; ++i) {
            row_data[i] /= sum;
        }
    }
};
```

### Solution 2: Quantized GEMM
```cpp
#include <cutlass/integer_subbyte.h>

// INT8 data type with proper CUTLASS integration
using int8_t_cutlass = cutlass::int8_t;

// Quantization epilogue
template<typename ElementOutput, typename ElementAccumulator>
class QuantizationEpilogue {
private:
    float alpha_scale_;
    float beta_scale_;
    float output_scale_;
    int8_t output_zero_point_;
    
public:
    using ElementCompute = float;  // Use float for scale factors
    using ElementAccumulatorInOut = ElementAccumulator;
    
    struct Arguments {
        float alpha_scale;
        float beta_scale;
        float output_scale;
        int8_t output_zero_point;
    };

    CUTLASS_HOST
    Status initialize(Arguments const &args) {
        alpha_scale_ = args.alpha_scale;
        beta_scale_ = args.beta_scale;
        output_scale_ = args.output_scale;
        output_zero_point_ = args.output_zero_point;
        return Status::kSuccess;
    }

    CUTLASS_HOST_DEVICE
    QuantizationEpilogue(Arguments const &args) {
        initialize(args);
    }

    CUTLASS_HOST_DEVICE
    QuantizationEpilogue(
        float alpha_scale = 1.0f,
        float beta_scale = 1.0f,
        float output_scale = 1.0f,
        int8_t output_zero_point = 0) :
        alpha_scale_(alpha_scale),
        beta_scale_(beta_scale),
        output_scale_(output_scale),
        output_zero_point_(output_zero_point) {}

    CUTLASS_HOST_DEVICE
    Array<int8_t_cutlass, 1> operator()(Array<ElementAccumulator, 1> const &frag_C,
                                       Array<ElementAccumulator, 1> const &frag_accum) const {
        Array<int8_t_cutlass, 1> frag_D;
        
        float intermediate = alpha_scale_ * float(frag_accum[0]) + 
                           beta_scale_ * float(frag_C[0]);
        
        // Quantize to INT8
        float dequantized = intermediate / output_scale_ + output_zero_point_;
        int8_t quantized = static_cast<int8_t>(round(dequantized));
        
        // Clamp to INT8 range
        quantized = max(static_cast<int8_t>(-128), min(static_cast<int8_t>(127), quantized));
        
        frag_D[0] = static_cast<int8_t_cutlass>(quantized);
        return frag_D;
    }
};

// INT8 GEMM wrapper
template<>
class CustomInt8Gemm {
public:
    using GemmType = cutlass::gemm::device::Gemm<
        int8_t_cutlass, cutlass::layout::ColumnMajor,
        int8_t_cutlass, cutlass::layout::ColumnMajor,
        int8_t_cutlass, cutlass::layout::ColumnMajor,
        int32_t  // Accumulator type for INT8
    >;
    
    using EpilogueOp = QuantizationEpilogue<int8_t_cutlass, int32_t>;
    
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        int8_t_cutlass const *ptr_A;
        int8_t_cutlass const *ptr_B;
        int8_t_cutlass const *ptr_C;
        int8_t_cutlass *ptr_D;
        typename EpilogueOp::Arguments epilogue_args;
    };
    
    Status operator()(Arguments const &args) {
        typename GemmType::Arguments gemm_args{
            args.problem_size,
            {args.ptr_A, args.problem_size.m(), args.problem_size.k()},
            {args.ptr_B, args.problem_size.k(), args.problem_size.n()},
            {args.ptr_C, args.problem_size.m(), args.problem_size.n()},
            {args.ptr_D, args.problem_size.m(), args.problem_size.n()},
            typename EpilogueOp::Params(args.epilogue_args)
        };
        
        GemmType gemm_op;
        return gemm_op(gemm_args);
    }
};
```

### Solution 3: Performance Tuning Framework
```cpp
#include <map>
#include <vector>

// Performance tuning framework
class CutlassTuner {
private:
    struct ProblemKey {
        int m, n, k;
        std::string data_type;
        std::string layout;
        
        bool operator<(const ProblemKey& other) const {
            if (m != other.m) return m < other.m;
            if (n != other.n) return n < other.n;
            if (k != other.k) return k < other.k;
            if (data_type != other.data_type) return data_type < other.data_type;
            return layout < other.layout;
        }
    };
    
    struct TunedConfig {
        cutlass::gemm::GemmCoord threadblock_shape;
        int stages;
        float performance_gflops;
        std::chrono::steady_clock::time_point timestamp;
    };
    
    static std::map<ProblemKey, TunedConfig> config_cache_;
    static const int CACHE_EXPIRY_SECONDS = 3600; // 1 hour
    
public:
    template<typename GemmOperator>
    static typename GemmOperator::Arguments find_best_configuration(
        typename GemmOperator::Arguments const &base_args) {
        
        ProblemKey key = {
            base_args.problem_size.m(),
            base_args.problem_size.n(), 
            base_args.problem_size.k(),
            "float",  // Would need to extract from template
            "column_major"  // Would need to extract from template
        };
        
        // Check cache first
        auto cache_it = config_cache_.find(key);
        if (cache_it != config_cache_.end()) {
            auto now = std::chrono::steady_clock::now();
            if (std::chrono::duration_cast<std::chrono::seconds>(
                    now - cache_it->second.timestamp).count() < CACHE_EXPIRY_SECONDS) {
                // Use cached configuration
                return modify_arguments(base_args, cache_it->second);
            }
        }
        
        // Benchmark different configurations
        std::vector<TunedConfig> configs = generate_candidate_configs(base_args.problem_size);
        TunedConfig best_config = benchmark_configs<GemmOperator>(base_args, configs);
        
        // Cache the result
        config_cache_[key] = best_config;
        
        return modify_arguments(base_args, best_config);
    }
    
private:
    static std::vector<TunedConfig> generate_candidate_configs(cutlass::gemm::GemmCoord problem_size) {
        std::vector<TunedConfig> configs;
        
        // Generate candidates based on problem size
        std::vector<cutlass::gemm::GemmShape> candidate_shapes = {
            {128, 128, 32}, {128, 256, 32}, {256, 128, 32},
            {64, 128, 32}, {128, 64, 32}, {64, 64, 32}
        };
        
        std::vector<int> stage_counts = {2, 3, 4};
        
        for (const auto& shape : candidate_shapes) {
            for (int stages : stage_counts) {
                if (problem_size.m() % shape.m() == 0 && 
                    problem_size.n() % shape.n() == 0 && 
                    problem_size.k() % shape.k() == 0) {
                    
                    TunedConfig config;
                    config.threadblock_shape = shape;
                    config.stages = stages;
                    configs.push_back(config);
                }
            }
        }
        
        return configs;
    }
    
    template<typename GemmOperator>
    static TunedConfig benchmark_configs(
        typename GemmOperator::Arguments const &base_args,
        std::vector<TunedConfig> const &configs) {
        
        TunedConfig best_config;
        float best_performance = 0.0f;
        
        for (const auto& config : configs) {
            typename GemmOperator::Arguments test_args = modify_arguments(base_args, config);
            
            float perf = benchmark_single_config<GemmOperator>(test_args);
            
            if (perf > best_performance) {
                best_performance = perf;
                best_config = config;
                best_config.performance_gflops = perf;
                best_config.timestamp = std::chrono::steady_clock::now();
            }
        }
        
        return best_config;
    }
    
    template<typename GemmOperator>
    static float benchmark_single_config(typename GemmOperator::Arguments const &args) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        GemmOperator op;
        
        // Warm up
        op(args);
        cudaDeviceSynchronize();
        
        // Time multiple runs
        cudaEventRecord(start);
        for (int i = 0; i < 5; ++i) {  // Multiple runs for stability
            op(args);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        float avg_time_ms = milliseconds / 5.0f;
        float total_ops = 2.0f * args.problem_size.m() * args.problem_size.n() * args.problem_size.k();
        float gflops = (total_ops / 1e9) / (avg_time_ms / 1000.0f);
        
        return gflops;
    }
    
    template<typename GemmOperator>
    static typename GemmOperator::Arguments modify_arguments(
        typename GemmOperator::Arguments const &base_args,
        TunedConfig const &config) {
        
        // This is a simplified example - in reality, you'd need to reconstruct
        // the entire operator with the new configuration
        auto modified_args = base_args;
        // Would need to recreate the operator with new template parameters
        return modified_args;
    }
};

// Static member definition
std::map<CutlassTuner::ProblemKey, CutlassTuner::TunedConfig> CutlassTuner::config_cache_;
```

## Advanced Topic: Integration with Deep Learning Frameworks

### PyTorch Integration Example
```cpp
// Example of integrating CUTLASS with PyTorch
#ifdef WITH_PYTORCH
#include <torch/extension.h>

torch::Tensor cutlass_gemm_wrapper(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor C,
    float alpha,
    float beta) {
    
    // Validate input tensors
    TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
    TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
    TORCH_CHECK(C.is_cuda(), "C must be a CUDA tensor");
    
    TORCH_CHECK(A.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(B.dtype() == torch::kFloat32, "Only float32 supported");
    TORCH_CHECK(C.dtype() == torch::kFloat32, "Only float32 supported");
    
    // Get tensor dimensions
    int m = A.size(0);
    int k = A.size(1);
    int n = B.size(1);
    
    TORCH_CHECK(B.size(0) == k, "A and B dimensions don't match");
    TORCH_CHECK(C.size(0) == m && C.size(1) == n, "C dimensions don't match");
    
    // Create CUTLASS tensors
    cutlass::gemm::GemmCoord problem_size(m, n, k);
    
    using Element = float;
    using Layout = cutlass::layout::ColumnMajor;
    
    cutlass::TensorRef<Element, Layout> ref_A(A.data_ptr<Element>(), m);
    cutlass::TensorRef<Element, Layout> ref_B(B.data_ptr<Element>(), k);
    cutlass::TensorRef<Element, Layout> ref_C(C.data_ptr<Element>(), m);
    
    // Allocate output tensor
    auto D = torch::empty({m, n}, torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCUDA));
    cutlass::TensorRef<Element, Layout> ref_D(D.data_ptr<Element>(), m);
    
    // Create GEMM operator
    using GemmOperator = cutlass::gemm::device::Gemm<Element, Layout, Element, Layout, Element, Layout, Element>;
    GemmOperator gemm_op;
    
    // Prepare arguments
    typename GemmOperator::Arguments args{
        problem_size,
        ref_A,
        ref_B,
        ref_C,
        ref_D,
        {alpha, beta}
    };
    
    // Initialize and run
    auto status = gemm_op.initialize(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to initialize CUTLASS GEMM");
    
    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "Failed to run CUTLASS GEMM");
    
    return D;
}
#endif
```

## Quiz Questions

1. How do custom epilogues improve performance in CUTLASS?

2. What are the key considerations when adding support for non-standard data types?

3. How can convolution operations be expressed as GEMM using im2col?

4. What factors influence the choice of threadblock shape in performance tuning?

5. How does the tuning framework cache and reuse optimal configurations?

## Summary
Module 8 covered advanced CUTLASS customization techniques, including custom epilogues, non-standard data types, tensor operations beyond GEMM, performance tuning strategies, debugging techniques, and integration with deep learning frameworks. These advanced topics enable users to extend CUTLASS for specialized applications and achieve optimal performance.