# Module 9: Real-world Applications and Case Studies

## Overview
This module explores practical applications of CUTLASS in real-world scenarios, including deep learning framework integration, quantized operations, sparse computations, mixed precision, and performance optimization techniques used in production environments.

## Learning Objectives
By the end of this module, students will be able to:
- Integrate CUTLASS with deep learning frameworks like PyTorch and TensorFlow
- Implement quantized matrix multiplication operations
- Work with sparse matrix operations using CUTLASS
- Apply mixed precision computation techniques
- Optimize memory bandwidth and occupancy
- Address numerical accuracy considerations in production

## Topic 1: Deep Learning Framework Integration

### PyTorch Integration
```cpp
// CUTLASS integration with PyTorch
#include <torch/extension.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

// CUTLASS headers
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/numeric_types.h>

torch::Tensor cutlass_gemm_fp16(
    torch::Tensor A,
    torch::Tensor B,
    torch::Tensor bias = torch::Tensor(),
    float alpha = 1.0f,
    float beta = 0.0f) {
    
    // Validate inputs
    TORCH_CHECK(A.is_cuda() && B.is_cuda(), "Inputs must be CUDA tensors");
    TORCH_CHECK(A.dtype() == torch::kHalf && B.dtype() == torch::kHalf, "Only FP16 supported");
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2, "Only 2D tensors supported");
    
    // Extract dimensions
    const int M = A.size(0);
    const int K = A.size(1);
    const int N = B.size(1);
    
    TORCH_CHECK(B.size(0) == K, "A and B dimensions don't match");
    
    // Create output tensor
    auto options = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCUDA);
    auto C = torch::zeros({M, N}, options);
    
    // Create CUTLASS tensors
    cutlass::gemm::GemmCoord problem_size(M, N, K);
    
    using Element = cutlass::half_t;
    using Layout = cutlass::layout::ColumnMajor;
    
    cutlass::TensorRef<Element, Layout> ref_A(
        static_cast<Element*>(A.data_ptr()), M);
    cutlass::TensorRef<Element, Layout> ref_B(
        static_cast<Element*>(B.data_ptr()), K);
    cutlass::TensorRef<Element, Layout> ref_C(
        static_cast<Element*>(C.data_ptr()), M);
    cutlass::TensorRef<Element, Layout> ref_D(
        static_cast<Element*>(C.data_ptr()), M);
    
    // Create GEMM operator
    using GemmOperator = cutlass::gemm::device::Gemm<
        Element, Layout,
        Element, Layout,
        Element, Layout,
        Element>;
    
    GemmOperator gemm_op;
    
    // Prepare arguments with bias epilogue if provided
    typename GemmOperator::Arguments args;
    
    if (bias.defined()) {
        // Use custom epilogue with bias addition
        using EpilogueOp = cutlass::epilogue::thread::LinearCombinationBiasElementwise<
            Element, 128 / cutlass::sizeof_bits<Element>::value, Element, Element>;
        
        typename EpilogueOp::Params epilogue_params{
            alpha, beta,
            static_cast<Element*>(bias.data_ptr())
        };
        
        args = typename GemmOperator::Arguments{
            problem_size,
            ref_A, ref_B, ref_C, ref_D,
            epilogue_params
        };
    } else {
        // Standard linear combination
        args = typename GemmOperator::Arguments{
            problem_size,
            ref_A, ref_B, ref_C, ref_D,
            {alpha, beta}
        };
    }
    
    // Initialize and run
    auto status = gemm_op.initialize(args);
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM initialization failed");
    
    status = gemm_op();
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS GEMM execution failed");
    
    return C;
}

// Registration for PyTorch
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("cutlass_gemm_fp16", &cutlass_gemm_fp16, "CUTLASS FP16 GEMM");
}
```

### TensorFlow Integration
```cpp
// TensorFlow custom op using CUTLASS
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

class CutlassGemmOp : public OpKernel {
public:
    explicit CutlassGemmOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
        OP_REQUIRES_OK(ctx, ctx->GetAttr("alpha", &alpha_));
        OP_REQUIRES_OK(ctx, ctx->GetAttr("beta", &beta_));
    }

    void Compute(OpKernelContext* ctx) override {
        // Get input tensors
        const Tensor& A = ctx->input(0);
        const Tensor& B = ctx->input(1);
        const Tensor& C = ctx->input(2);  // Bias or previous result
        
        // Validate input shapes
        OP_REQUIRES(ctx, A.dims() == 2 && B.dims() == 2,
                    errors::InvalidArgument("A and B must be 2D"));
        OP_REQUIRES(ctx, A.dim_size(1) == B.dim_size(0),
                    errors::InvalidArgument("A and B dimensions don't match"));
        
        // Extract dimensions
        int M = A.dim_size(0);
        int K = A.dim_size(1);
        int N = B.dim_size(1);
        
        // Create output tensor
        Tensor* output = nullptr;
        OP_REQUIRES_OK(ctx, ctx->allocate_output(0, TensorShape{M, N}, &output));
        
        // Call CUTLASS GEMM
        auto A_flat = A.flat<float>();
        auto B_flat = B.flat<float>();
        auto C_flat = C.flat<float>();
        auto output_flat = output->flat<float>();
        
        run_cutlass_gemm(
            A_flat.data(), B_flat.data(), C_flat.data(), output_flat.mutable_data(),
            M, N, K, alpha_, beta_);
    }

private:
    float alpha_;
    float beta_;
    
    void run_cutlass_gemm(
        const float* A, const float* B, const float* C, float* D,
        int M, int N, int K, float alpha, float beta) {
        
        cutlass::gemm::GemmCoord problem_size(M, N, K);
        
        using Element = float;
        using Layout = cutlass::layout::ColumnMajor;
        
        cutlass::TensorRef<Element, Layout> ref_A(A, M);
        cutlass::TensorRef<Element, Layout> ref_B(B, K);
        cutlass::TensorRef<Element, Layout> ref_C(C, M);
        cutlass::TensorRef<Element, Layout> ref_D(D, M);
        
        using GemmOperator = cutlass::gemm::device::Gemm<Element, Layout, Element, Layout, Element, Layout, Element>;
        GemmOperator gemm_op;
        
        typename GemmOperator::Arguments args{
            problem_size,
            ref_A, ref_B, ref_C, ref_D,
            {alpha, beta}
        };
        
        auto status = gemm_op.initialize(args);
        if (status != cutlass::Status::kSuccess) {
            // Handle error appropriately
            return;
        }
        
        status = gemm_op();
        if (status != cutlass::Status::kSuccess) {
            // Handle error appropriately
            return;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("CutlassGemm").Device(DEVICE_GPU), CutlassGemmOp);
```

## Topic 2: Quantized Matrix Multiplication

### INT8 Quantized GEMM
```cpp
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/quantization/quantization.h>

// Quantized GEMM with INT8 inputs and INT32 accumulator
class QuantizedInt8Gemm {
public:
    using ElementInputA = cutlass::int8_t;
    using ElementInputB = cutlass::int8_t;
    using ElementOutput = cutlass::int32_t;
    using ElementAccumulator = cutlass::int32_t;
    
    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    
    using GemmOperator = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator
    >;
    
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        ElementInputA const *ptr_A;
        ElementInputB const *ptr_B;
        ElementOutput const *ptr_C;
        ElementOutput *ptr_D;
        float alpha;  // Scale factor for A*B
        float beta;   // Scale factor for C
        float scale_A; // Quantization scale for A
        float scale_B; // Quantization scale for B
        float scale_D; // Quantization scale for output
    };
    
    Status operator()(Arguments const &args) {
        // Create tensor references
        cutlass::TensorRef<ElementInputA, LayoutA> ref_A(args.ptr_A, args.problem_size.m());
        cutlass::TensorRef<ElementInputB, LayoutB> ref_B(args.ptr_B, args.problem_size.k());
        cutlass::TensorRef<ElementOutput, LayoutC> ref_C(args.ptr_C, args.problem_size.m());
        cutlass::TensorRef<ElementOutput, LayoutC> ref_D(args.ptr_D, args.problem_size.m());
        
        // Create epilogue with quantization parameters
        using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementOutput>;
        
        typename EpilogueOp::Params epilogue_params{
            args.alpha * args.scale_A * args.scale_B / args.scale_D,  // Combined scale
            args.beta,
            ElementOutput(1 << 24),  // Clamp threshold
            ElementOutput(-(1 << 24))
        };
        
        typename GemmOperator::Arguments gemm_args{
            args.problem_size,
            ref_A, ref_B, ref_C, ref_D,
            epilogue_params
        };
        
        GemmOperator gemm_op;
        return gemm_op(gemm_args);
    }
};

// Quantization utilities
class QuantizationUtils {
public:
    // Quantize float tensor to INT8
    static void quantize_tensor(
        float const *input, cutlass::int8_t *output, 
        int size, float scale, int8_t zero_point = 0) {
        
        for (int i = 0; i < size; ++i) {
            float scaled = input[i] / scale + zero_point;
            int8_t quantized = static_cast<int8_t>(roundf(scaled));
            // Clamp to [-128, 127]
            quantized = std::max(static_cast<int8_t>(-128), 
                               std::min(static_cast<int8_t>(127), quantized));
            output[i] = quantized;
        }
    }
    
    // Dequantize INT8 tensor to float
    static void dequantize_tensor(
        cutlass::int8_t const *input, float *output,
        int size, float scale, int8_t zero_point = 0) {
        
        for (int i = 0; i < size; ++i) {
            float dequantized = (input[i] - zero_point) * scale;
            output[i] = dequantized;
        }
    }
    
    // Calculate optimal scale for quantization
    static float calculate_quantization_scale(float const *data, int size) {
        float max_abs = 0.0f;
        for (int i = 0; i < size; ++i) {
            max_abs = std::max(max_abs, fabsf(data[i]));
        }
        
        // Use 127 as maximum representable value for INT8
        return max_abs / 127.0f;
    }
};
```

### Dynamic Quantization Example
```cpp
// Dynamic quantization wrapper
template<typename FloatType, typename IntType>
class DynamicQuantizationGemm {
private:
    QuantizedInt8Gemm quantized_gemm_;
    
public:
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        FloatType const *ptr_A;
        FloatType const *ptr_B;
        FloatType const *ptr_C;
        FloatType *ptr_D;
        bool per_channel_quantization;
    };
    
    Status operator()(Arguments const &args) {
        // Calculate quantization scales dynamically
        float scale_A = QuantizationUtils::calculate_quantization_scale(
            args.ptr_A, args.problem_size.m() * args.problem_size.k());
        float scale_B = QuantizationUtils::calculate_quantization_scale(
            args.ptr_B, args.problem_size.k() * args.problem_size.n());
        
        // Allocate temporary quantized tensors
        size_t size_A = args.problem_size.m() * args.problem_size.k();
        size_t size_B = args.problem_size.k() * args.problem_size.n();
        size_t size_C = args.problem_size.m() * args.problem_size.n();
        
        cutlass::int8_t *quantized_A, *quantized_B;
        cutlass::int32_t *quantized_C, *quantized_D;
        
        cudaMalloc(&quantized_A, size_A * sizeof(cutlass::int8_t));
        cudaMalloc(&quantized_B, size_B * sizeof(cutlass::int8_t));
        cudaMalloc(&quantized_C, size_C * sizeof(cutlass::int32_t));
        cudaMalloc(&quantized_D, size_C * sizeof(cutlass::int32_t));
        
        // Quantize inputs
        QuantizationUtils::quantize_tensor(args.ptr_A, quantized_A, size_A, scale_A);
        QuantizationUtils::quantize_tensor(args.ptr_B, quantized_B, size_B, scale_B);
        QuantizationUtils::quantize_tensor(args.ptr_C, quantized_C, size_C, 1.0f); // No scaling for bias
        
        // Run quantized GEMM
        auto quantized_args = QuantizedInt8Gemm::Arguments{
            args.problem_size,
            quantized_A, quantized_B, quantized_C, quantized_D,
            1.0f, 0.0f,  // alpha, beta
            scale_A, scale_B, 1.0f  // scales
        };
        
        Status status = quantized_gemm_(quantized_args);
        
        // Dequantize output
        float output_scale = scale_A * scale_B;
        QuantizationUtils::dequantize_tensor(quantized_D, args.ptr_D, size_C, output_scale);
        
        // Cleanup
        cudaFree(quantized_A);
        cudaFree(quantized_B);
        cudaFree(quantized_C);
        cudaFree(quantized_D);
        
        return status;
    }
};
```

## Topic 3: Sparse Operations

### Sparse Matrix Dense Matrix Multiplication (SpMM)
```cpp
#include <cutlass/gemm/device/gemm_sparse.h>

// CUTLASS sparse GEMM implementation
class CutlassSparseGemm {
public:
    using ElementA = cutlass::half_t;  // Sparse weights
    using ElementB = cutlass::half_t;  // Dense activations
    using ElementC = cutlass::half_t;  // Output
    using ElementAccumulator = float;
    
    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    
    using GemmOperator = cutlass::gemm::device::GemmSparse<
        ElementA, LayoutA,
        ElementB, LayoutB,
        ElementC, LayoutC,
        ElementAccumulator
    >;
    
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        ElementA const *ptr_A;           // Sparse weight matrix
        uint1_ptr const *ptr_B;          // Meta data for sparse matrix
        ElementB const *ptr_B;           // Dense input matrix
        ElementC const *ptr_C;           // Bias/input
        ElementC *ptr_D;                 // Output
        int meta_ld;                     // Leading dimension of metadata
        float alpha;
        float beta;
    };
    
    Status operator()(Arguments const &args) {
        cutlass::TensorRef<ElementA, LayoutA> ref_A(args.ptr_A, args.problem_size.m());
        cutlass::TensorRef<uint1_ptr, cutlass::layout::ColumnMajor> ref_meta(args.ptr_meta, args.meta_ld);
        cutlass::TensorRef<ElementB, LayoutB> ref_B(args.ptr_B, args.problem_size.k());
        cutlass::TensorRef<ElementC, LayoutC> ref_C(args.ptr_C, args.problem_size.m());
        cutlass::TensorRef<ElementC, LayoutC> ref_D(args.ptr_D, args.problem_size.m());
        
        typename GemmOperator::Arguments gemm_args{
            args.problem_size,
            ref_A, ref_meta, ref_B, ref_C, ref_D,
            {args.alpha, args.beta}
        };
        
        GemmOperator gemm_op;
        return gemm_op(gemm_args);
    }
};

// Sparsity pattern creation utilities
class SparsityPattern {
public:
    // Create 2:4 structured sparsity pattern
    static void create_2to4_sparsity(
        float const *dense_weights,
        cutlass::half_t *sparse_weights,
        uint1_ptr *metadata,
        int rows, int cols) {
        
        for (int col = 0; col < cols; col += 4) {
            for (int row = 0; row < rows; ++row) {
                // For each group of 4 elements in a row, keep 2 with highest magnitude
                float values[4] = {0, 0, 0, 0};
                bool selected[4] = {false, false, false, false};
                
                // Get 4 values
                for (int k = 0; k < 4 && col + k < cols; ++k) {
                    values[k] = dense_weights[row * cols + col + k];
                }
                
                // Find indices of 2 largest absolute values
                std::vector<std::pair<float, int>> indexed_values;
                for (int k = 0; k < 4; ++k) {
                    indexed_values.push_back({fabsf(values[k]), k});
                }
                
                std::sort(indexed_values.rbegin(), indexed_values.rend());
                
                // Mark top 2 as selected
                for (int k = 0; k < 2; ++k) {
                    if (indexed_values[k].second < 4) {
                        selected[indexed_values[k].second] = true;
                    }
                }
                
                // Store sparse values and metadata
                for (int k = 0; k < 4 && col + k < cols; ++k) {
                    if (selected[k]) {
                        sparse_weights[row * cols + col + k] = 
                            cutlass::half_t(dense_weights[row * cols + col + k]);
                    } else {
                        sparse_weights[row * cols + col + k] = cutlass::half_t(0);
                    }
                }
                
                // Update metadata (implementation depends on CUTLASS version)
                // This is a simplified representation
            }
        }
    }
};
```

## Topic 4: Mixed Precision Computations

### FP16 Accumulation in FP32
```cpp
// Mixed precision GEMM: FP16 inputs, FP32 accumulation, FP16 output
class MixedPrecisionGemm {
public:
    using ElementInputA = cutlass::half_t;
    using ElementInputB = cutlass::half_t;
    using ElementOutput = cutlass::half_t;
    using ElementAccumulator = float;  // Higher precision accumulator
    
    using LayoutA = cutlass::layout::ColumnMajor;
    using LayoutB = cutlass::layout::ColumnMajor;
    using LayoutC = cutlass::layout::ColumnMajor;
    
    using GemmOperator = cutlass::gemm::device::Gemm<
        ElementInputA, LayoutA,
        ElementInputB, LayoutB,
        ElementOutput, LayoutC,
        ElementAccumulator
    >;
    
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        ElementInputA const *ptr_A;
        ElementInputB const *ptr_B;
        ElementOutput const *ptr_C;
        ElementOutput *ptr_D;
        float alpha;
        float beta;
    };
    
    Status operator()(Arguments const &args) {
        cutlass::TensorRef<ElementInputA, LayoutA> ref_A(args.ptr_A, args.problem_size.m());
        cutlass::TensorRef<ElementInputB, LayoutB> ref_B(args.ptr_B, args.problem_size.k());
        cutlass::TensorRef<ElementOutput, LayoutC> ref_C(args.ptr_C, args.problem_size.m());
        cutlass::TensorRef<ElementOutput, LayoutC> ref_D(args.ptr_D, args.problem_size.m());
        
        // Use linear combination with clamping for numerical stability
        using EpilogueOp = cutlass::epilogue::thread::LinearCombinationClamp<
            ElementOutput, 128 / cutlass::sizeof_bits<ElementOutput>::value,
            ElementAccumulator, ElementOutput>;
        
        typename EpilogueOp::Params epilogue_params{
            args.alpha, args.beta,
            ElementOutput(65504.0_hf),   // Max FP16 value
            ElementOutput(-65504.0_hf)   // Min FP16 value
        };
        
        typename GemmOperator::Arguments gemm_args{
            args.problem_size,
            ref_A, ref_B, ref_C, ref_D,
            epilogue_params
        };
        
        GemmOperator gemm_op;
        return gemm_op(gemm_args);
    }
};
```

### Tensor Core Mixed Precision
```cpp
// Tensor Core optimized mixed precision
class TensorCoreMixedPrecision {
public:
    // For Volta and newer architectures with Tensor Cores
    using ElementA = cutlass::half_t;
    using ElementB = cutlass::half_t;
    using ElementC = cutlass::half_t;
    using ElementAccumulator = float;  // FP32 accumulator for Tensor Cores
    
    using OperatorClass = cutlass::arch::OpClassTensorOp;
    using ArchTag = cutlass::arch::Sm80;  // Or appropriate architecture
    
    using DefaultGemm = cutlass::gemm::device::DefaultGemmConfiguration<
        OperatorClass, ArchTag, ElementA, ElementB, ElementC, ElementAccumulator>;
    
    using GemmOperator = typename DefaultGemm::GemmKernel;
    
    struct Arguments {
        cutlass::gemm::GemmCoord problem_size;
        ElementA const *ptr_A;
        ElementB const *ptr_B;
        ElementC const *ptr_C;
        ElementC *ptr_D;
        float alpha;
        float beta;
    };
    
    Status operator()(Arguments const &args) {
        // Create tensor references
        cutlass::TensorRef<ElementA, typename DefaultGemm::LayoutA> ref_A(args.ptr_A, args.problem_size.m());
        cutlass::TensorRef<ElementB, typename DefaultGemm::LayoutB> ref_B(args.ptr_B, args.problem_size.k());
        cutlass::TensorRef<ElementC, typename DefaultGemm::LayoutC> ref_C(args.ptr_C, args.problem_size.m());
        cutlass::TensorRef<ElementC, typename DefaultGemm::LayoutC> ref_D(args.ptr_D, args.problem_size.m());
        
        // Create epilogue parameters
        typename DefaultGemm::EpilogueOutputOp::Params epilogue_params{args.alpha, args.beta};
        
        typename GemmOperator::Arguments gemm_args{
            args.problem_size,
            ref_A, ref_B, ref_C, ref_D,
            epilogue_params
        };
        
        // Initialize and run
        GemmOperator gemm_op;
        return gemm_op(gemm_args);
    }
};
```

## Topic 5: Memory Bandwidth and Occupancy Optimization

### Memory Access Optimization
```cpp
// Optimized memory access patterns
class MemoryOptimizedGemm {
public:
    // Use vectorized loads for better memory throughput
    template<int VectorSize = 4>
    struct VectorizedLoader {
        CUTLASS_DEVICE
        static void load(cutlass::Array<float, VectorSize> &frag, float const *ptr, int offset) {
            reinterpret_cast<float4*>(&frag)[0] = reinterpret_cast<float4 const*>(ptr)[offset];
        }
        
        CUTLASS_DEVICE
        static void store(float *ptr, int offset, cutlass::Array<float, VectorSize> const &frag) {
            reinterpret_cast<float4*>(ptr)[offset] = reinterpret_cast<float4 const&>(frag);
        }
    };
    
    // Shared memory tiling with padding to avoid bank conflicts
    template<int Rows, int Cols, typename Element>
    struct PaddedSharedMemory {
        // Add padding to avoid bank conflicts (typically +1 element per 32)
        static constexpr int kPaddedCols = Cols + (Cols % 32 == 0 ? 1 : 0);
        Element storage[Rows][kPaddedCols];
        
        CUTLASS_DEVICE
        Element& access(int row, int col) {
            return storage[row][col];
        }
    };
    
    // Coalesced memory access pattern
    template<typename Element, typename Layout>
    struct CoalescedAccess {
        CUTLASS_DEVICE
        static int get_address(int thread_id, int element_offset, int stride) {
            if constexpr (std::is_same_v<Layout, cutlass::layout::ColumnMajor>) {
                // For column-major, consecutive threads access consecutive rows
                return thread_id * stride + element_offset;
            } else {
                // For row-major, consecutive threads access consecutive columns
                return element_offset * stride + thread_id;
            }
        }
    };
};
```

### Occupancy Optimization
```cpp
// Occupancy optimization utilities
class OccupancyOptimizer {
public:
    // Calculate optimal block size for maximum occupancy
    static int calculate_optimal_block_size(
        void const *kernel_func,
        size_t dynamic_smem_bytes,
        int max_threads_per_sm) {
        
        int min_grid_size, block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &block_size,
            reinterpret_cast<const void*>(kernel_func),
            dynamic_smem_bytes, 0);
        
        return block_size;
    }
    
    // Calculate theoretical occupancy
    static float calculate_theoretical_occupancy(
        void const *kernel_func,
        int block_size,
        size_t dynamic_smem_bytes) {
        
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks,
            reinterpret_cast<const void*>(kernel_func),
            block_size, dynamic_smem_bytes);
        
        float theoretical_occupancy = 
            static_cast<float>(max_active_blocks * block_size) / prop.maxThreadsPerMultiProcessor;
        
        return theoretical_occupancy;
    }
    
    // Occupancy-optimized kernel launch
    template<typename GemmOperator>
    static Status launch_with_optimization(
        typename GemmOperator::Arguments const &args) {
        
        GemmOperator gemm_op;
        
        // Calculate optimal configuration
        int optimal_block_size = calculate_optimal_block_size(
            reinterpret_cast<void const*>(&gemm_op),
            0,  // dynamic shared memory
            args.problem_size.m() * args.problem_size.n() / 1000  // estimate
        );
        
        // Initialize with optimal parameters
        auto status = gemm_op.initialize(args);
        if (status != cutlass::Status::kSuccess) {
            return status;
        }
        
        // Run the optimized kernel
        return gemm_op();
    }
};
```

## Topic 6: Numerical Accuracy Considerations

### Precision Analysis and Error Bounds
```cpp
// Numerical accuracy analysis tools
class NumericalAccuracyAnalyzer {
public:
    // Compare CUTLASS result with reference implementation
    static float calculate_max_relative_error(
        float const *cutlass_result,
        float const *reference_result,
        int size) {
        
        float max_error = 0.0f;
        for (int i = 0; i < size; ++i) {
            float ref_val = reference_result[i];
            float cutlass_val = cutlass_result[i];
            
            float absolute_error = fabsf(ref_val - cutlass_val);
            float relative_error = (ref_val != 0.0f) ? 
                                  absolute_error / fabsf(ref_val) : absolute_error;
                                  
            max_error = std::max(max_error, relative_error);
        }
        
        return max_error;
    }
    
    // Calculate ULP (Units in Last Place) error
    static int calculate_max_ulp_error(
        float const *cutlass_result,
        float const *reference_result,
        int size) {
        
        int max_ulp = 0;
        for (int i = 0; i < size; ++i) {
            uint32_t ref_bits = *reinterpret_cast<uint32_t const*>(&reference_result[i]);
            uint32_t cut_bits = *reinterpret_cast<uint32_t const*>(&cutlass_result[i]);
            
            // Handle different signs
            if ((ref_bits >> 31) != (cut_bits >> 31)) {
                // Different signs, calculate from zero
                uint32_t ref_abs = ref_bits & 0x7FFFFFFF;
                uint32_t cut_abs = cut_bits & 0x7FFFFFFF;
                int ulp = static_cast<int>(std::max(ref_abs, cut_abs));
                max_ulp = std::max(max_ulp, ulp);
            } else {
                // Same signs, calculate difference
                int ulp = static_cast<int>(abs(static_cast<int32_t>(ref_bits) - static_cast<int32_t>(cut_bits)));
                max_ulp = std::max(max_ulp, ulp);
            }
        }
        
        return max_ulp;
    }
    
    // Statistical error analysis
    struct ErrorStats {
        float max_relative_error;
        float mean_relative_error;
        float rms_error;
        float variance;
    };
    
    static ErrorStats compute_error_statistics(
        float const *cutlass_result,
        float const *reference_result,
        int size) {
        
        ErrorStats stats{};
        double sum_rel_errors = 0.0;
        double sum_sq_errors = 0.0;
        float max_error = 0.0f;
        
        for (int i = 0; i < size; ++i) {
            float ref_val = reference_result[i];
            float cutlass_val = cutlass_result[i];
            
            if (ref_val != 0.0f) {
                float rel_error = fabsf((ref_val - cutlass_val) / ref_val);
                max_error = std::max(max_error, rel_error);
                sum_rel_errors += rel_error;
                sum_sq_errors += rel_error * rel_error;
            }
        }
        
        stats.max_relative_error = max_error;
        stats.mean_relative_error = static_cast<float>(sum_rel_errors / size);
        stats.rms_error = sqrtf(static_cast<float>(sum_sq_errors / size));
        
        // Calculate variance
        double sum_sq_diffs = 0.0;
        for (int i = 0; i < size; ++i) {
            float ref_val = reference_result[i];
            if (ref_val != 0.0f) {
                float rel_error = fabsf((ref_val - cutlass_result[i]) / ref_val);
                double diff = rel_error - stats.mean_relative_error;
                sum_sq_diffs += diff * diff;
            }
        }
        stats.variance = static_cast<float>(sum_sq_diffs / size);
        
        return stats;
    }
};
```

## Hands-on Exercises

### Exercise 1: Transformer Attention Implementation
Implement a multi-head attention mechanism using CUTLASS.

```cpp
// TODO: Implement multi-head attention using CUTLASS:
// 1. Query-Key matmul for attention scores
// 2. Apply softmax to attention scores
// 3. Score-Value matmul for output
// 4. Handle batched operations
// 5. Optimize for transformer-specific dimensions
```

### Exercise 2: Quantized Neural Network Layer
Create a complete quantized linear layer using CUTLASS.

```cpp
// TODO: Create a quantized linear layer that:
// 1. Quantizes inputs and weights to INT8
// 2. Performs INT8 GEMM using CUTLASS
// 3. Dequantizes output back to FP32
// 4. Includes proper scaling and zero-point handling
// 5. Integrates with a deep learning framework
```

### Exercise 3: Performance Comparison Study
Compare CUTLASS performance with cuBLAS for different scenarios.

```cpp
// TODO: Create a performance study that:
// 1. Benchmarks CUTLASS vs cuBLAS for various matrix sizes
// 2. Tests different data types (FP32, FP16, INT8)
// 3. Measures GFLOPS, bandwidth, and latency
// 4. Analyzes performance differences
// 5. Provides recommendations for when to use each
```

## Solutions to Exercises

### Solution 1: Transformer Attention Implementation
```cpp
#include <cutlass/gemm/device/gemm.h>

class TransformerAttention {
public:
    using ElementType = cutlass::half_t;
    using Layout = cutlass::layout::ColumnMajor;
    
    using GemmOperator = cutlass::gemm::device::Gemm<
        ElementType, Layout,
        ElementType, Layout,
        ElementType, Layout,
        ElementType>;
    
    struct Arguments {
        int batch_size;
        int seq_len;
        int head_dim;
        int num_heads;
        
        // Input tensors: [batch_size, num_heads, seq_len, head_dim]
        ElementType const *query;    // Q
        ElementType const *key;      // K  
        ElementType const *value;    // V
        ElementType *attention_scores;  // Output of Q*K^T
        ElementType *output;         // Output of attention*V
    };
    
    Status compute_attention(Arguments const &args) {
        Status status = cutlass::Status::kSuccess;
        
        // Step 1: Compute Q * K^T for attention scores
        // Q: [seq_len, head_dim], K: [seq_len, head_dim]
        // Result: [seq_len, seq_len] (for each head and batch)
        cutlass::gemm::GemmCoord score_problem_size(
            args.seq_len, args.seq_len, args.head_dim);
        
        for (int batch = 0; batch < args.batch_size; ++batch) {
            for (int head = 0; head < args.num_heads; ++head) {
                // Calculate offsets for current batch and head
                int qkv_offset = (batch * args.num_heads + head) * args.seq_len * args.head_dim;
                int score_offset = (batch * args.num_heads + head) * args.seq_len * args.seq_len;
                
                // Q * K^T: [seq_len, head_dim] * [head_dim, seq_len] = [seq_len, seq_len]
                cutlass::TensorRef<ElementType, Layout> ref_Q(
                    args.query + qkv_offset, args.seq_len);
                cutlass::TensorRef<ElementType, Layout> ref_K(
                    args.key + qkv_offset, args.seq_len);
                cutlass::TensorRef<ElementType, Layout> ref_scores(
                    args.attention_scores + score_offset, args.seq_len);
                
                // Need to transpose K, so swap operands and layouts
                typename GemmOperator::Arguments score_args{
                    score_problem_size,
                    ref_Q, ref_K, ref_scores, ref_scores,  // Use scores as both C and D for simplicity
                    {ElementType(1.0f), ElementType(0.0f)}
                };
                
                GemmOperator gemm_op;
                status = gemm_op(score_args);
                if (status != cutlass::Status::kSuccess) return status;
                
                // Apply scaling (typically 1/sqrt(head_dim))
                scale_attention_scores(
                    args.attention_scores + score_offset, 
                    args.seq_len * args.seq_len, 
                    ElementType(1.0f / sqrtf(float(args.head_dim))));
                
                // Step 2: Apply softmax to attention scores
                apply_softmax(
                    args.attention_scores + score_offset,
                    args.seq_len, args.seq_len);
                
                // Step 3: Compute attention_scores * V
                // [seq_len, seq_len] * [seq_len, head_dim] = [seq_len, head_dim]
                cutlass::gemm::GemmCoord output_problem_size(
                    args.seq_len, args.head_dim, args.seq_len);
                
                cutlass::TensorRef<ElementType, Layout> ref_V(
                    args.value + qkv_offset, args.seq_len);
                cutlass::TensorRef<ElementType, Layout> ref_output(
                    args.output + qkv_offset, args.seq_len);
                
                typename GemmOperator::Arguments output_args{
                    output_problem_size,
                    {args.attention_scores + score_offset, args.seq_len},  // Scores
                    ref_V,  // Values
                    ref_output, ref_output,  // Output
                    {ElementType(1.0f), ElementType(0.0f)}
                };
                
                status = gemm_op(output_args);
                if (status != cutlass::Status::kSuccess) return status;
            }
        }
        
        return status;
    }

private:
    void scale_attention_scores(ElementType* scores, int size, ElementType scale) {
        for (int i = 0; i < size; ++i) {
            scores[i] = scores[i] * scale;
        }
    }
    
    void apply_softmax(ElementType* matrix, int rows, int cols) {
        for (int i = 0; i < rows; ++i) {
            // Find max in row for numerical stability
            ElementType max_val = matrix[i * cols];
            for (int j = 1; j < cols; ++j) {
                if (matrix[i * cols + j] > max_val) {
                    max_val = matrix[i * cols + j];
                }
            }
            
            // Compute exp and sum
            ElementType sum = ElementType(0);
            for (int j = 0; j < cols; ++j) {
                ElementType exp_val = exp(matrix[i * cols + j] - max_val);
                matrix[i * cols + j] = exp_val;
                sum = sum + exp_val;
            }
            
            // Normalize
            for (int j = 0; j < cols; ++j) {
                matrix[i * cols + j] = matrix[i * cols + j] / sum;
            }
        }
    }
};
```

### Solution 2: Quantized Neural Network Layer
```cpp
#include <cutlass/gemm/device/gemm.h>

class QuantizedLinearLayer {
private:
    // Quantized INT8 GEMM operator
    using ElementInput = cutlass::int8_t;
    using ElementWeight = cutlass::int8_t;
    using ElementOutput = cutlass::int32_t;
    using ElementAccumulator = cutlass::int32_t;
    
    using Layout = cutlass::layout::ColumnMajor;
    
    using GemmOperator = cutlass::gemm::device::Gemm<
        ElementWeight, Layout,
        ElementInput, Layout,
        ElementOutput, Layout,
        ElementAccumulator>;
    
    // Storage for quantized weights
    ElementWeight *d_quantized_weights_;
    int input_features_;
    int output_features_;
    
    // Quantization parameters
    float weight_scale_;
    float input_scale_;
    float output_scale_;
    
public:
    QuantizedLinearLayer(int input_features, int output_features) 
        : input_features_(input_features), output_features_(output_features),
          d_quantized_weights_(nullptr), weight_scale_(1.0f), 
          input_scale_(1.0f), output_scale_(1.0f) {}
    
    ~QuantizedLinearLayer() {
        if (d_quantized_weights_) {
            cudaFree(d_quantized_weights_);
        }
    }
    
    Status initialize(float const *fp32_weights) {
        // Calculate weight scale
        weight_scale_ = calculate_quantization_scale(fp32_weights, input_features_ * output_features_);
        
        // Quantize weights
        std::vector<ElementWeight> h_quantized_weights(input_features_ * output_features_);
        quantize_tensor(fp32_weights, h_quantized_weights.data(), 
                       input_features_ * output_features_, weight_scale_);
        
        // Allocate device memory for quantized weights
        cudaMalloc(&d_quantized_weights_, 
                  input_features_ * output_features_ * sizeof(ElementWeight));
        
        // Copy quantized weights to device
        cudaMemcpy(d_quantized_weights_, h_quantized_weights.data(),
                  input_features_ * output_features_ * sizeof(ElementWeight),
                  cudaMemcpyHostToDevice);
        
        return cutlass::Status::kSuccess;
    }
    
    Status forward(
        ElementInput const *input,      // Quantized input [batch_size, input_features]
        ElementOutput *output,          // Quantized output [batch_size, output_features]
        int batch_size) {
        
        cutlass::gemm::GemmCoord problem_size(
            output_features_, batch_size, input_features_);
        
        cutlass::TensorRef<ElementWeight, Layout> ref_weight(
            d_quantized_weights_, output_features_);
        cutlass::TensorRef<ElementInput, Layout> ref_input(
            input, input_features_);
        cutlass::TensorRef<ElementOutput, Layout> ref_output(
            output, output_features_);
        
        // Use identity for C (no bias in this example)
        typename GemmOperator::Arguments args{
            problem_size,
            ref_weight, ref_input, ref_output, ref_output,
            {ElementAccumulator(1), ElementAccumulator(0)}
        };
        
        GemmOperator gemm_op;
        return gemm_op(args);
    }
    
    // Dequantize output to FP32
    void dequantize_output(ElementOutput const *quantized_output, 
                          float *fp32_output, int size) {
        float combined_scale = weight_scale_ * input_scale_ / output_scale_;
        
        for (int i = 0; i < size; ++i) {
            fp32_output[i] = static_cast<float>(quantized_output[i]) * combined_scale;
        }
    }

private:
    float calculate_quantization_scale(float const *data, int size) {
        float max_abs = 0.0f;
        for (int i = 0; i < size; ++i) {
            max_abs = std::max(max_abs, fabsf(data[i]));
        }
        return max_abs / 127.0f;  // INT8 range is [-128, 127]
    }
    
    void quantize_tensor(float const *input, ElementWeight *output, 
                        int size, float scale) {
        for (int i = 0; i < size; ++i) {
            float scaled = input[i] / scale;
            int8_t quantized = static_cast<int8_t>(roundf(scaled));
            quantized = std::max(static_cast<int8_t>(-128), 
                               std::min(static_cast<int8_t>(127), quantized));
            output[i] = static_cast<ElementWeight>(quantized);
        }
    }
};
```

### Solution 3: Performance Comparison Study
```cpp
#include <chrono>
#include <vector>
#include <iostream>

class PerformanceComparison {
public:
    struct BenchmarkResult {
        float gflops;
        float bandwidth_gbs;
        float latency_ms;
        float efficiency_percent;
    };
    
    struct ComparisonReport {
        int m, n, k;
        BenchmarkResult cutlass_result;
        BenchmarkResult cublas_result;
        float speedup;  // CUTLASS vs cuBLAS
    };
    
    static std::vector<ComparisonReport> run_comparison_study() {
        std::vector<ComparisonReport> reports;
        
        // Test various matrix sizes
        std::vector<std::tuple<int, int, int>> test_sizes = {
            {512, 512, 512},    // Small
            {1024, 1024, 1024}, // Medium
            {2048, 2048, 2048}, // Large
            {4096, 4096, 4096}, // Very large
            {8192, 2048, 2048}, // Rectangular
            {2048, 8192, 2048},
        };
        
        for (auto [m, n, k] : test_sizes) {
            ComparisonReport report;
            report.m = m;
            report.n = n;
            report.k = k;
            
            // Benchmark CUTLASS
            report.cutlass_result = benchmark_cutlass<float>(m, n, k);
            
            // Benchmark cuBLAS
            report.cublas_result = benchmark_cublas<float>(m, n, k);
            
            // Calculate speedup
            report.speedup = report.cublas_result.latency_ms / report.cutlass_result.latency_ms;
            
            reports.push_back(report);
        }
        
        return reports;
    }
    
    static void print_comparison_report(std::vector<ComparisonReport> const &reports) {
        std::cout << "Matrix Size\tCUTLASS GFLOPS\tcuBLAS GFLOPS\tSpeedup\tRecommendation\n";
        std::cout << "----------\t--------------\t-----------\t-------\t-------------\n";
        
        for (auto const &report : reports) {
            std::string recommendation = 
                (report.speedup > 1.1f) ? "Use CUTLASS" :
                (report.speedup < 0.9f) ? "Use cuBLAS" :
                "Similar performance";
                
            std::cout << report.m << "x" << report.n << "x" << report.k << "\t\t"
                      << report.cutlass_result.gflops << "\t\t"
                      << report.cublas_result.gflops << "\t\t"
                      << report.speedup << "x\t"
                      << recommendation << "\n";
        }
    }

private:
    template<typename Element>
    static BenchmarkResult benchmark_cutlass(int m, int n, int k) {
        // Allocate memory
        Element *A, *B, *C, *D;
        cudaMalloc(&A, m * k * sizeof(Element));
        cudaMalloc(&B, k * n * sizeof(Element));
        cudaMalloc(&C, m * n * sizeof(Element));
        cudaMalloc(&D, m * n * sizeof(Element));
        
        // Initialize with random data
        initialize_random_data(A, m * k);
        initialize_random_data(B, k * n);
        initialize_random_data(C, m * n);
        
        // Create CUTLASS GEMM
        using Layout = cutlass::layout::ColumnMajor;
        using GemmOperator = cutlass::gemm::device::Gemm<Element, Layout, Element, Layout, Element, Layout, Element>;
        
        cutlass::gemm::GemmCoord problem_size(m, n, k);
        cutlass::TensorRef<Element, Layout> ref_A(A, m);
        cutlass::TensorRef<Element, Layout> ref_B(B, k);
        cutlass::TensorRef<Element, Layout> ref_C(C, m);
        cutlass::TensorRef<Element, Layout> ref_D(D, m);
        
        typename GemmOperator::Arguments args{
            problem_size,
            ref_A, ref_B, ref_C, ref_D,
            {Element(1), Element(0)}
        };
        
        GemmOperator gemm_op;
        gemm_op.initialize(args);
        
        // Warm up
        gemm_op();
        cudaDeviceSynchronize();
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        const int num_runs = 10;
        for (int i = 0; i < num_runs; ++i) {
            gemm_op();
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        float total_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        float avg_time_ms = total_time_ms / num_runs;
        
        BenchmarkResult result;
        result.latency_ms = avg_time_ms;
        result.gflops = (2.0f * m * n * k) / (avg_time_ms * 1e6f);
        result.bandwidth_gbs = (sizeof(Element) * (m*k + k*n + m*n)) / (avg_time_ms * 1e6f);
        result.efficiency_percent = (result.gflops / get_peak_gflops()) * 100.0f;
        
        // Cleanup
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(D);
        
        return result;
    }
    
    template<typename Element>
    static BenchmarkResult benchmark_cublas(int m, int n, int k) {
        // cuBLAS benchmark implementation
        cublasHandle_t handle;
        cublasCreate(&handle);
        
        Element *A, *B, *C, *D;
        cudaMalloc(&A, m * k * sizeof(Element));
        cudaMalloc(&B, k * n * sizeof(Element));
        cudaMalloc(&C, m * n * sizeof(Element));
        cudaMalloc(&D, m * n * sizeof(Element));
        
        initialize_random_data(A, m * k);
        initialize_random_data(B, k * n);
        initialize_random_data(C, m * n);
        
        Element alpha = Element(1), beta = Element(0);
        
        // Warm up
        if constexpr (std::is_same_v<Element, float>) {
            cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k, &alpha, A, m, B, k, &beta, C, m);
        } else if constexpr (std::is_same_v<Element, double>) {
            cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k, &alpha, A, m, B, k, &beta, C, m);
        } else if constexpr (std::is_same_v<Element, cutlass::half_t>) {
            cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                       m, n, k, 
                       reinterpret_cast<__half*>(&alpha), 
                       reinterpret_cast<__half*>(A), m,
                       reinterpret_cast<__half*>(B), k,
                       reinterpret_cast<__half*>(&beta),
                       reinterpret_cast<__half*>(C), m);
        }
        cudaDeviceSynchronize();
        
        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();
        const int num_runs = 10;
        for (int i = 0; i < num_runs; ++i) {
            if constexpr (std::is_same_v<Element, float>) {
                cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           m, n, k, &alpha, A, m, B, k, &beta, C, m);
            } else if constexpr (std::is_same_v<Element, double>) {
                cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           m, n, k, &alpha, A, m, B, k, &beta, C, m);
            } else if constexpr (std::is_same_v<Element, cutlass::half_t>) {
                cublasHgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                           m, n, k,
                           reinterpret_cast<__half*>(&alpha),
                           reinterpret_cast<__half*>(A), m,
                           reinterpret_cast<__half*>(B), k,
                           reinterpret_cast<__half*>(&beta),
                           reinterpret_cast<__half*>(C), m);
            }
        }
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        
        float total_time_ms = std::chrono::duration<float, std::milli>(end - start).count();
        float avg_time_ms = total_time_ms / num_runs;
        
        BenchmarkResult result;
        result.latency_ms = avg_time_ms;
        result.gflops = (2.0f * m * n * k) / (avg_time_ms * 1e6f);
        result.bandwidth_gbs = (sizeof(Element) * (m*k + k*n + m*n)) / (avg_time_ms * 1e6f);
        result.efficiency_percent = (result.gflops / get_peak_gflops()) * 100.0f;
        
        // Cleanup
        cudaFree(A);
        cudaFree(B);
        cudaFree(C);
        cudaFree(D);
        cublasDestroy(handle);
        
        return result;
    }
    
    template<typename Element>
    static void initialize_random_data(Element* data, int size) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(-1.0f, 1.0f);
        
        std::vector<float> temp(size);
        for (int i = 0; i < size; ++i) {
            temp[i] = dis(gen);
        }
        
        cudaMemcpy(data, temp.data(), size * sizeof(Element), cudaMemcpyHostToDevice);
    }
    
    static float get_peak_gflops() {
        // Return theoretical peak for the current GPU
        // This would be determined based on the GPU compute capability
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        // Simplified calculation (would need to be more accurate in practice)
        return prop.memoryClockRate * prop.memoryBusWidth / 8 * 2 / 1e6; // Approximate
    }
};
```

## Advanced Topic: Production Deployment Considerations

### Memory Management in Production
```cpp
// Production-ready memory management for CUTLASS
class ProductionMemoryManager {
private:
    struct MemoryPool {
        void* ptr;
        size_t size;
        bool in_use;
        std::chrono::steady_clock::time_point last_used;
    };
    
    std::vector<MemoryPool> pools_;
    std::mutex mutex_;
    const size_t kMaxPoolSize = 1024 * 1024 * 1024; // 1GB max per pool
    
public:
    void* allocate(size_t size) {
        std::lock_guard<std::mutex> lock(mutex_);
        
        // Try to find reusable memory
        for (auto& pool : pools_) {
            if (!pool.in_use && pool.size >= size) {
                pool.in_use = true;
                pool.last_used = std::chrono::steady_clock::now();
                return pool.ptr;
            }
        }
        
        // Allocate new memory if needed
        if (get_total_allocated() + size <= kMaxPoolSize) {
            void* ptr;
            cudaMalloc(&ptr, size);
            pools_.push_back({ptr, size, true, std::chrono::steady_clock::now()});
            return ptr;
        }
        
        // If we can't allocate, try to free old unused memory
        cleanup_old_allocations();
        
        // Retry allocation
        void* ptr;
        cudaMalloc(&ptr, size);
        pools_.push_back({ptr, size, true, std::chrono::steady_clock::now()});
        return ptr;
    }
    
    void release(void* ptr) {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& pool : pools_) {
            if (pool.ptr == ptr) {
                pool.in_use = false;
                pool.last_used = std::chrono::steady_clock::now();
                break;
            }
        }
    }
    
private:
    size_t get_total_allocated() {
        size_t total = 0;
        for (const auto& pool : pools_) {
            if (pool.in_use) {
                total += pool.size;
            }
        }
        return total;
    }
    
    void cleanup_old_allocations() {
        auto now = std::chrono::steady_clock::now();
        for (auto it = pools_.begin(); it != pools_.end();) {
            // Free memory not used in the last 10 seconds
            if (!it->in_use && 
                std::chrono::duration_cast<std::chrono::seconds>(now - it->last_used).count() > 10) {
                cudaFree(it->ptr);
                it = pools_.erase(it);
            } else {
                ++it;
            }
        }
    }
};
```

## Quiz Questions

1. What are the main advantages of integrating CUTLASS with deep learning frameworks?

2. How does quantization affect the accuracy and performance of neural networks?

3. What is structured sparsity and how does it benefit GEMM operations?

4. Explain the benefits of mixed precision computation in neural networks.

5. How do you measure and compare the numerical accuracy of different GEMM implementations?

## Summary
Module 9 explored real-world applications of CUTLASS including deep learning framework integration, quantized operations, sparse computations, mixed precision techniques, and production deployment considerations. These practical examples demonstrate how CUTLASS is used in production environments for high-performance computing applications.