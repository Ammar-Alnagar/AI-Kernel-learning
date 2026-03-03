# Module 10: Performance Optimization and Profiling

## Overview
This module covers advanced performance analysis and optimization techniques for CUTLASS, including profiling tools, memory bandwidth optimization, occupancy maximization, and numerical accuracy considerations in production environments.

## Learning Objectives
By the end of this module, students will be able to:
- Use GPU profiling tools (Nsight Compute, nvprof) effectively
- Optimize memory bandwidth utilization
- Maximize occupancy for different problem sizes
- Optimize register usage and cache efficiency
- Implement asynchronous operations for better performance
- Analyze and improve numerical accuracy in production code

## Topic 1: GPU Profiling Tools

### Nsight Compute Profiling
```cuda
#include <cuda_runtime.h>
#include <nvtx3/nvToolsExt.h>

// Profiling utilities for CUTLASS operations
class NsightProfile {
public:
    struct ProfileMetrics {
        float achieved_occupancy;
        float sm_efficiency;
        float dram_read_throughput;
        float dram_write_throughput;
        float tensor_core_utilization;
        float stall_reasons[10];  // Various stall reasons
    };
    
    static void profile_kernel_launch(
        std::function<void()> kernel_func,
        const char* kernel_name,
        ProfileMetrics& metrics) {
        
        // Mark region for profiling
        nvtxRangePush(kernel_name);
        
        // Launch kernel
        kernel_func();
        cudaDeviceSynchronize();
        
        // In practice, you would use NCU command line or API to collect metrics
        // This is a simplified representation
        collect_metrics(kernel_name, metrics);
        
        nvtxRangePop();
    }
    
    static void collect_metrics(const char* kernel_name, ProfileMetrics& metrics) {
        // This would typically call Nsight Compute API or parse profiler output
        // For demonstration, we'll simulate metric collection
        
        // Simulated metrics (in practice, these come from profiler)
        metrics.achieved_occupancy = 0.85f;  // 85% occupancy
        metrics.sm_efficiency = 0.92f;      // 92% SM efficiency
        metrics.dram_read_throughput = 850.0f;  // GB/s
        metrics.dram_write_throughput = 420.0f; // GB/s
        metrics.tensor_core_utilization = 0.78f; // 78% utilization
    }
    
    // Profile memory access patterns
    static void profile_memory_access(
        void* ptr, size_t size, 
        const char* description) {
        
        printf("Memory profiling for %s:\n", description);
        printf("  Address: %p\n", ptr);
        printf("  Size: %zu bytes\n", size);
        printf("  Alignment: %zu bytes\n", 
               reinterpret_cast<uintptr_t>(ptr) % 128);
        
        // Check for coalescing opportunities
        size_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr % 128 == 0) {
            printf("  ✓ Aligned to 128-byte boundary\n");
        } else {
            printf("  ⚠ Not aligned to 128-byte boundary\n");
        }
    }
};

// Example usage of profiling
void profile_cutlass_gemm() {
    NsightProfile::ProfileMetrics metrics;
    
    // Wrap CUTLASS kernel launch with profiling
    auto kernel_func = []() {
        // CUTLASS GEMM kernel launch would go here
        // For example:
        // cutlass_gemm_op(args);
    };
    
    NsightProfile::profile_kernel_launch(
        kernel_func, "cutlass_gemm_fp16", metrics);
    
    printf("Profiling Results:\n");
    printf("  Achieved Occupancy: %.2f%%\n", metrics.achieved_occupancy * 100);
    printf("  SM Efficiency: %.2f%%\n", metrics.sm_efficiency * 100);
    printf("  DRAM Read Throughput: %.1f GB/s\n", metrics.dram_read_throughput);
    printf("  DRAM Write Throughput: %.1f GB/s\n", metrics.dram_write_throughput);
    printf("  Tensor Core Utilization: %.2f%%\n", metrics.tensor_core_utilization * 100);
}
```

### Custom Profiling Macros
```cuda
// Custom profiling macros for CUTLASS
#define PROFILE_START(name) \
    cudaEvent_t start_##name, stop_##name; \
    cudaEventCreate(&start_##name); \
    cudaEventCreate(&stop_##name); \
    cudaEventRecord(start_##name);

#define PROFILE_END(name, desc) \
    cudaEventRecord(stop_##name); \
    cudaEventSynchronize(stop_##name); \
    float time_##name; \
    cudaEventElapsedTime(&time_##name, start_##name, stop_##name); \
    printf("%s took %.3f ms\n", desc, time_##name); \
    cudaEventDestroy(start_##name); \
    cudaEventDestroy(stop_##name);

// Usage example
void profile_cutlass_operation() {
    PROFILE_START(gemm_compute);
    
    // CUTLASS GEMM operation
    // cutlass_gemm_op(args);
    
    PROFILE_END(gemm_compute, "CUTLASS GEMM");
}
```

## Topic 2: Memory Bandwidth Optimization

### Memory Access Pattern Analysis
```cuda
// Memory bandwidth optimization utilities
class MemoryBandwidthOptimizer {
public:
    // Analyze memory access patterns
    struct MemoryAnalysis {
        float theoretical_bandwidth_gb_s;
        float achieved_bandwidth_gb_s;
        float bandwidth_utilization_percent;
        bool has_coalesced_access;
        int bank_conflicts_per_access;
    };
    
    static MemoryAnalysis analyze_memory_access(
        int problem_size_m, int problem_size_n, int problem_size_k,
        int data_type_size_bytes) {
        
        MemoryAnalysis analysis;
        
        // Calculate theoretical memory bandwidth
        // This depends on the GPU's memory specifications
        float peak_bandwidth_gb_s = get_gpu_peak_bandwidth();
        
        // Calculate required memory operations
        size_t bytes_read = (problem_size_m * problem_size_k + 
                           problem_size_k * problem_size_n) * data_type_size_bytes;
        size_t bytes_written = problem_size_m * problem_size_n * data_type_size_bytes;
        size_t total_bytes = bytes_read + bytes_written;
        
        // Estimate computation time based on peak TFLOPS
        float peak_tflops = get_gpu_peak_tflops();
        float ops = 2.0f * problem_size_m * problem_size_n * problem_size_k;  // 2 ops per multiply-add
        float estimated_time_s = (ops / 1e12f) / peak_tflops;  // Time at peak performance
        
        analysis.theoretical_bandwidth_gb_s = (total_bytes / 1e9f) / estimated_time_s;
        analysis.achieved_bandwidth_gb_s = 0; // Would come from profiler
        analysis.bandwidth_utilization_percent = 
            (analysis.achieved_bandwidth_gb_s / peak_bandwidth_gb_s) * 100.0f;
        
        // Check for coalescing (simplified check)
        analysis.has_coalesced_access = (problem_size_n % 32 == 0); // Multiple of warp size
        
        // Bank conflict estimation
        analysis.bank_conflicts_per_access = estimate_bank_conflicts(problem_size_m, problem_size_n);
        
        return analysis;
    }
    
    // Optimize memory access for better bandwidth
    template<typename Element, typename Layout>
    static void optimize_memory_layout(
        Element* input, Element* output,
        int rows, int cols) {
        
        // For column-major layout, ensure coalesced access
        if constexpr (std::is_same_v<Layout, cutlass::layout::ColumnMajor>) {
            // Process in tiles to improve cache locality
            const int TILE_SIZE = 64;
            
            for (int i = 0; i < rows; i += TILE_SIZE) {
                for (int j = 0; j < cols; j += TILE_SIZE) {
                    // Process tile [i:i+TILE_SIZE, j:j+TILE_SIZE]
                    for (int ii = i; ii < min(i + TILE_SIZE, rows); ++ii) {
                        for (int jj = j; jj < min(j + TILE_SIZE, cols); ++jj) {
                            // Access is coalesced for column-major when processing by columns
                            output[ii * cols + jj] = input[ii * cols + jj];
                        }
                    }
                }
            }
        }
    }
    
    // Shared memory optimization
    template<int Rows, int Cols, typename Element>
    struct OptimizedSharedMemory {
        // Add padding to avoid bank conflicts
        static constexpr int kPaddedCols = Cols + (Cols % 32 == 0 ? 1 : 0);
        Element data[Rows][kPaddedCols];
        
        CUTLASS_DEVICE
        Element& access(int row, int col) {
            return data[row][col];
        }
        
        CUTLASS_DEVICE
        void load_from_global(Element const* global_ptr, int global_ld, int block_row, int block_col) {
            int local_row = threadIdx.y;
            int local_col = threadIdx.x;
            
            if (block_row + local_row < Rows && block_col + local_col < Cols) {
                data[local_row][local_col] = 
                    global_ptr[(block_row + local_row) * global_ld + (block_col + local_col)];
            }
        }
    };

private:
    static float get_gpu_peak_bandwidth() {
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        // Simplified calculation (real calculation is more complex)
        return (prop.memoryClockRate * 1e3f * prop.memoryBusWidth / 8) / 1e9f;
    }
    
    static float get_gpu_peak_tflops() {
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&device);
        
        // Simplified calculation for FP32
        return (prop.multiProcessorCount * 2 * prop.clockRate) / 1e9f; // Rough estimate
    }
    
    static int estimate_bank_conflicts(int rows, int cols) {
        // Simplified bank conflict estimation
        // In practice, this would be more sophisticated
        if (cols % 32 == 0) {
            return 32; // Worst case: all threads access same bank
        }
        return 1; // Best case: no conflicts
    }
};
```

### Memory Prefetching and Streaming
```cuda
// Memory prefetching utilities
class MemoryPrefetcher {
public:
    // Prefetch data to different memory spaces
    template<typename Element>
    static CUTLASS_DEVICE void prefetch_L1(Element const* ptr) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        asm volatile("prefetch.global.L1 [%0];" : : "l"(ptr));
        #endif
    }
    
    template<typename Element>
    static CUTLASS_DEVICE void prefetch_L2(Element const* ptr) {
        #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 350
        asm volatile("prefetch.global.L2 [%0];" : : "l"(ptr));
        #endif
    }
    
    // Stream-based memory operations
    static void async_memory_operations() {
        cudaStream_t stream1, stream2;
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        
        // Example: overlap computation with memory transfers
        // Compute on stream1 while transferring data on stream2
        // This is conceptual - actual implementation would depend on specific use case
    }
};
```

## Topic 3: Occupancy Optimization

### Occupancy Analysis and Optimization
```cuda
// Occupancy optimization utilities
class OccupancyOptimizer {
public:
    struct OccupancyInfo {
        int max_active_blocks_per_sm;
        int theoretical_occupancy_percent;
        int actual_occupancy_percent;
        int limiting_resource;  // 0=blocks, 1=smem, 2=registers
        size_t shared_memory_per_block;
        int registers_per_thread;
    };
    
    static OccupancyInfo analyze_occupancy(
        void const* kernel_func,
        int block_size_x, int block_size_y, int block_size_z,
        size_t dynamic_smem_bytes) {
        
        OccupancyInfo info{};
        
        int device;
        cudaGetDevice(&device);
        
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, device);
        
        int min_grid_size, optimal_block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &optimal_block_size,
            reinterpret_cast<const void*>(kernel_func),
            dynamic_smem_bytes, 0);
        
        int max_active_blocks;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks,
            reinterpret_cast<const void*>(kernel_func),
            optimal_block_size, dynamic_smem_bytes);
        
        info.max_active_blocks_per_sm = max_active_blocks;
        info.theoretical_occupancy_percent = 
            (max_active_blocks * optimal_block_size * 100) / prop.maxThreadsPerMultiProcessor;
        
        // Determine limiting resource
        int max_blocks_by_smem = prop.sharedMemPerMultiprocessor / dynamic_smem_bytes;
        int max_blocks_by_registers = prop.regsPerMultiprocessor / 
                                     (info.registers_per_thread * optimal_block_size);
        
        if (max_blocks_by_smem <= max_blocks_by_registers) {
            info.limiting_resource = 1; // Shared memory
        } else if (max_blocks_by_registers <= max_blocks_by_smem) {
            info.limiting_resource = 2; // Registers
        } else {
            info.limiting_resource = 0; // Blocks
        }
        
        return info;
    }
    
    // Optimize block size for maximum occupancy
    static int find_optimal_block_size(
        void const* kernel_func,
        size_t dynamic_smem_bytes) {
        
        int min_grid_size, optimal_block_size;
        cudaOccupancyMaxPotentialBlockSize(
            &min_grid_size, &optimal_block_size,
            reinterpret_cast<const void*>(kernel_func),
            dynamic_smem_bytes, 0);
        
        return optimal_block_size;
    }
    
    // Occupancy-aware kernel launch
    template<typename KernelFunc>
    static void launch_with_optimal_occupancy(
        KernelFunc kernel,
        dim3 grid_dim,
        size_t dynamic_smem_bytes = 0) {
        
        // Find optimal block size
        int optimal_block_size = find_optimal_block_size(
            reinterpret_cast<const void*>(&kernel), dynamic_smem_bytes);
        
        // Convert to 2D/3D block dimensions if needed
        dim3 block_dim;
        if (optimal_block_size <= 1024) {
            block_dim = dim3(optimal_block_size, 1, 1);
        } else if (optimal_block_size <= 1024 * 1024) {
            block_dim = dim3(min(optimal_block_size, 1024), 
                           (optimal_block_size + 1023) / 1024, 1);
        } else {
            block_dim = dim3(1024, 1024, (optimal_block_size + 1024*1024 - 1) / (1024*1024));
        }
        
        // Launch kernel with optimal configuration
        kernel<<<grid_dim, block_dim, dynamic_smem_bytes>>>();
    }
    
    // Occupancy monitoring during execution
    static void monitor_occupancy_during_execution() {
        // In practice, this would use CUDA profiling APIs
        // to monitor occupancy during kernel execution
        
        // Example: Check occupancy at different phases
        cudaDeviceProp prop;
        int device;
        cudaGetDevice(&device);
        cudaGetDeviceProperties(&prop, device);
        
        printf("GPU %d: Max threads per SM: %d\n", device, prop.maxThreadsPerMultiProcessor);
        printf("GPU %d: Max blocks per SM: %d\n", device, prop.maxBlocksPerMultiProcessor);
        printf("GPU %d: Shared memory per SM: %zu bytes\n", device, prop.sharedMemPerMultiprocessor);
        printf("GPU %d: Registers per SM: %d\n", device, prop.regsPerMultiprocessor);
    }
};
```

### Register Usage Optimization
```cuda
// Register usage optimization
class RegisterOptimizer {
public:
    // Techniques to reduce register usage
    template<typename Element, int FragmentSize>
    struct RegisterFragment {
        Element data[FragmentSize];
        
        CUTLASS_DEVICE
        void clear() {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentSize; ++i) {
                data[i] = Element(0);
            }
        }
        
        CUTLASS_DEVICE
        RegisterFragment& operator+=(const RegisterFragment& rhs) {
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < FragmentSize; ++i) {
                data[i] += rhs.data[i];
            }
            return *this;
        }
    };
    
    // Use local arrays instead of individual variables when possible
    CUTLASS_DEVICE
    static void optimize_local_variables() {
        // Bad: Uses more registers
        // float a0 = ..., a1 = ..., a2 = ..., a3 = ...;
        
        // Good: Uses fewer registers
        float locals[4];
        // Use locals[0], locals[1], etc.
    }
    
    // Limit loop unrolling to control register usage
    CUTLASS_DEVICE
    static void controlled_unrolling(float* data, int size) {
        // Instead of full unrolling which increases register pressure
        // Use partial unrolling or no unrolling based on register constraints
        
        #ifdef CUTLASS_OPTIMIZED_REGISTERS
        // Limited unrolling for register-constrained scenarios
        for (int i = 0; i < size; i += 4) {
            if (i < size) data[i] *= 2.0f;
            if (i+1 < size) data[i+1] *= 2.0f;
            if (i+2 < size) data[i+2] *= 2.0f;
            if (i+3 < size) data[i+3] *= 2.0f;
        }
        #else
        // Full computation
        for (int i = 0; i < size; ++i) {
            data[i] *= 2.0f;
        }
        #endif
    }
};
```

## Topic 4: Asynchronous Operations

### CUDA Streams and Events
```cuda
// Asynchronous operations for better performance
class AsyncOperations {
private:
    std::vector<cudaStream_t> streams_;
    std::vector<cudaEvent_t> events_;
    int num_streams_;
    
public:
    AsyncOperations(int num_streams = 4) : num_streams_(num_streams) {
        streams_.resize(num_streams_);
        events_.resize(num_streams_);
        
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamCreate(&streams_[i]);
            cudaEventCreate(&events_[i]);
        }
    }
    
    ~AsyncOperations() {
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamDestroy(streams_[i]);
            cudaEventDestroy(events_[i]);
        }
    }
    
    // Asynchronous CUTLASS operations
    template<typename CutlassGemmOp>
    void async_gemm_batch(
        typename CutlassGemmOp::Arguments* args,
        int num_batches) {
        
        for (int i = 0; i < num_batches; ++i) {
            int stream_idx = i % num_streams_;
            
            // Record event before kernel launch
            cudaEventRecord(events_[stream_idx], streams_[stream_idx]);
            
            // Launch CUTLASS GEMM on specific stream
            CutlassGemmOp gemm_op;
            gemm_op.initialize(args[i]);
            gemm_op.run(streams_[stream_idx]);  // Assuming CUTLASS supports streams
            
            // Record event after kernel launch
            cudaEventRecord(events_[stream_idx], streams_[stream_idx]);
        }
        
        // Wait for all operations to complete
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
    }
    
    // Memory operations with streams
    void async_memory_copy(
        void* dst, const void* src, size_t count,
        int stream_idx = 0) {
        
        cudaStream_t stream = streams_[stream_idx % num_streams_];
        cudaMemcpyAsync(dst, src, count, cudaMemcpyDeviceToDevice, stream);
    }
    
    // Overlap computation with memory transfers
    void overlap_computation_with_transfer() {
        const int N = 1024 * 1024;
        const int chunk_size = N / num_streams_;
        
        float *d_input, *d_output, *d_intermediate;
        cudaMalloc(&d_input, N * sizeof(float));
        cudaMalloc(&d_output, N * sizeof(float));
        cudaMalloc(&d_intermediate, N * sizeof(float));
        
        for (int i = 0; i < num_streams_; ++i) {
            int offset = i * chunk_size;
            
            // Copy chunk to intermediate buffer
            cudaMemcpyAsync(
                d_intermediate + offset, 
                d_input + offset, 
                chunk_size * sizeof(float),
                cudaMemcpyDeviceToDevice, 
                streams_[i]);
            
            // Process chunk asynchronously
            process_chunk_async(
                d_intermediate + offset,
                d_output + offset,
                chunk_size,
                streams_[i]);
        }
        
        // Synchronize all streams
        for (int i = 0; i < num_streams_; ++i) {
            cudaStreamSynchronize(streams_[i]);
        }
        
        cudaFree(d_input);
        cudaFree(d_output);
        cudaFree(d_intermediate);
    }

private:
    void process_chunk_async(float* input, float* output, int size, cudaStream_t stream) {
        // Launch processing kernel asynchronously
        // process_kernel<<<blocks, threads, 0, stream>>>(input, output, size);
    }
};
```

### Event-Based Synchronization
```cuda
// Event-based synchronization for performance
class EventSynchronizer {
public:
    // Timing with events
    static float time_kernel_launch(std::function<void()> kernel_func) {
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        kernel_func();
        cudaEventRecord(stop);
        
        cudaEventSynchronize(stop);
        
        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        return milliseconds;
    }
    
    // Dependency management
    static void manage_dependencies() {
        cudaEvent_t event1, event2;
        cudaStream_t stream1, stream2;
        
        cudaEventCreate(&event1);
        cudaEventCreate(&event2);
        cudaStreamCreate(&stream1);
        cudaStreamCreate(&stream2);
        
        // Launch kernel on stream1
        // kernel1<<<blocks, threads, 0, stream1>>>();
        cudaEventRecord(event1, stream1);
        
        // Wait for event1 before launching on stream2
        cudaStreamWaitEvent(stream2, event1, 0);
        // kernel2<<<blocks, threads, 0, stream2>>>();
        
        cudaEventRecord(event2, stream2);
        
        // Wait for all operations
        cudaEventSynchronize(event2);
        
        cudaEventDestroy(event1);
        cudaEventDestroy(event2);
        cudaStreamDestroy(stream1);
        cudaStreamDestroy(stream2);
    }
};
```

## Topic 5: Numerical Accuracy in Production

### Accuracy Monitoring and Validation
```cuda
// Numerical accuracy validation tools
class AccuracyValidator {
public:
    // Compare results with reference implementation
    template<typename T>
    static float calculate_max_relative_error(
        T const* computed, 
        T const* reference, 
        int size,
        T tolerance = T(1e-5)) {
        
        float max_error = 0.0f;
        int error_count = 0;
        
        for (int i = 0; i < size; ++i) {
            T ref_val = reference[i];
            T comp_val = computed[i];
            
            if (ref_val != T(0)) {
                float rel_error = fabsf((ref_val - comp_val) / ref_val);
                max_error = fmaxf(max_error, rel_error);
                
                if (rel_error > tolerance) {
                    error_count++;
                }
            } else {
                float abs_error = fabsf(comp_val);
                max_error = fmaxf(max_error, abs_error);
                
                if (abs_error > tolerance) {
                    error_count++;
                }
            }
        }
        
        if (error_count > 0) {
            printf("Warning: %d/%d values exceed tolerance of %f\n", 
                   error_count, size, float(tolerance));
        }
        
        return max_error;
    }
    
    // ULP (Units in Last Place) error calculation
    static int calculate_max_ulp_error(float const* computed, float const* reference, int size) {
        int max_ulp = 0;
        
        for (int i = 0; i < size; ++i) {
            uint32_t ref_bits = *reinterpret_cast<uint32_t const*>(&reference[i]);
            uint32_t comp_bits = *reinterpret_cast<uint32_t const*>(&computed[i]);
            
            // Handle special cases
            if (isnan(reference[i]) && isnan(computed[i])) {
                continue; // Both NaN, consider equal
            }
            
            if (isinf(reference[i]) && isinf(computed[i]) && 
                signbit(reference[i]) == signbit(computed[i])) {
                continue; // Both infinity with same sign, consider equal
            }
            
            // Calculate ULP difference
            int32_t ulp_diff = abs(static_cast<int32_t>(ref_bits) - static_cast<int32_t>(comp_bits));
            max_ulp = max(max_ulp, ulp_diff);
        }
        
        return max_ulp;
    }
    
    // Statistical accuracy analysis
    struct AccuracyStats {
        float max_error;
        float mean_error;
        float rms_error;
        float variance;
        float percentile_95_error;
        int total_comparisons;
        int failed_comparisons;
        float failure_rate;
    };
    
    template<typename T>
    static AccuracyStats compute_accuracy_stats(
        T const* computed, 
        T const* reference, 
        int size,
        T tolerance = T(1e-5)) {
        
        AccuracyStats stats{};
        std::vector<float> errors;
        double sum_errors = 0.0;
        double sum_squared_errors = 0.0;
        
        for (int i = 0; i < size; ++i) {
            T ref_val = reference[i];
            T comp_val = computed[i];
            
            float error;
            if (ref_val != T(0)) {
                error = fabsf((ref_val - comp_val) / ref_val);
            } else {
                error = fabsf(comp_val);
            }
            
            errors.push_back(error);
            sum_errors += error;
            sum_squared_errors += error * error;
            
            if (error > tolerance) {
                stats.failed_comparisons++;
            }
        }
        
        stats.total_comparisons = size;
        stats.failure_rate = float(stats.failed_comparisons) / size;
        stats.max_error = *std::max_element(errors.begin(), errors.end());
        stats.mean_error = float(sum_errors / size);
        stats.rms_error = sqrtf(float(sum_squared_errors / size));
        
        // Calculate variance
        double sum_variance = 0.0;
        for (float error : errors) {
            double diff = error - stats.mean_error;
            sum_variance += diff * diff;
        }
        stats.variance = float(sum_variance / size);
        
        // Calculate 95th percentile
        std::sort(errors.begin(), errors.end());
        int percentile_95_idx = static_cast<int>(0.95 * errors.size());
        stats.percentile_95_error = errors[percentile_95_idx];
        
        return stats;
    }
    
    // Accuracy validation with logging
    template<typename T>
    static bool validate_accuracy_with_logging(
        T const* computed,
        T const* reference,
        int size,
        T tolerance = T(1e-5),
        const char* operation_name = "unknown") {
        
        AccuracyStats stats = compute_accuracy_stats(computed, reference, size, tolerance);
        
        printf("Accuracy validation for %s:\n", operation_name);
        printf("  Total comparisons: %d\n", stats.total_comparisons);
        printf("  Failed comparisons: %d (%.2f%%)\n", 
               stats.failed_comparisons, stats.failure_rate * 100);
        printf("  Max relative error: %e\n", stats.max_error);
        printf("  Mean relative error: %e\n", stats.mean_error);
        printf("  RMS error: %e\n", stats.rms_error);
        printf("  95th percentile error: %e\n", stats.percentile_95_error);
        
        bool is_accurate = stats.failure_rate <= 0.05f; // Less than 5% failures
        printf("  Result: %s\n", is_accurate ? "PASS" : "FAIL");
        
        return is_accurate;
    }
};
```

### Production Accuracy Monitoring
```cuda
// Production accuracy monitoring system
class ProductionAccuracyMonitor {
private:
    struct AccuracyRecord {
        std::string operation_name;
        int m, n, k;
        float tolerance;
        float max_error;
        float failure_rate;
        std::chrono::steady_clock::time_point timestamp;
        bool is_validated;
    };
    
    std::vector<AccuracyRecord> accuracy_log_;
    std::mutex log_mutex_;
    float global_tolerance_threshold_;
    
public:
    ProductionAccuracyMonitor(float tolerance = 1e-4f) 
        : global_tolerance_threshold_(tolerance) {}
    
    template<typename T>
    bool monitor_accuracy(
        T const* computed,
        T const* reference,
        int m, int n, int k,
        const char* operation_name,
        T tolerance = T(0)) {
        
        if (tolerance == T(0)) {
            tolerance = static_cast<T>(global_tolerance_threshold_);
        }
        
        // Calculate accuracy statistics
        auto stats = AccuracyValidator::compute_accuracy_stats(
            computed, reference, m * n, tolerance);
        
        // Log the result
        {
            std::lock_guard<std::mutex> lock(log_mutex_);
            AccuracyRecord record;
            record.operation_name = operation_name;
            record.m = m;
            record.n = n;
            record.k = k;
            record.tolerance = tolerance;
            record.max_error = stats.max_error;
            record.failure_rate = stats.failure_rate;
            record.timestamp = std::chrono::steady_clock::now();
            record.is_validated = stats.failure_rate <= 0.05f;
            
            accuracy_log_.push_back(record);
        }
        
        // Alert if accuracy is poor
        if (stats.failure_rate > 0.1f) {  // More than 10% failures
            printf("CRITICAL: Poor accuracy detected in %s: %f%% failures\n",
                   operation_name, stats.failure_rate * 100);
            return false;
        } else if (stats.failure_rate > 0.05f) {  // More than 5% failures
            printf("WARNING: Accuracy degradation detected in %s: %f%% failures\n",
                   operation_name, stats.failure_rate * 100);
        }
        
        return stats.failure_rate <= 0.05f;
    }
    
    // Get accuracy trends
    std::vector<AccuracyRecord> get_recent_accuracy_records(int count = 10) {
        std::lock_guard<std::mutex> lock(log_mutex_);
        
        if (accuracy_log_.size() <= count) {
            return accuracy_log_;
        }
        
        return std::vector<AccuracyRecord>(
            accuracy_log_.end() - count, accuracy_log_.end());
    }
    
    // Accuracy trend analysis
    struct TrendAnalysis {
        float avg_failure_rate;
        float max_failure_rate;
        float min_failure_rate;
        bool is_improving;
        bool is_degrading;
    };
    
    TrendAnalysis analyze_accuracy_trends() {
        auto recent_records = get_recent_accuracy_records(20);
        
        if (recent_records.empty()) {
            return {0, 0, 0, false, false};
        }
        
        float sum_failures = 0;
        float max_fail = 0;
        float min_fail = 1;
        
        for (const auto& record : recent_records) {
            sum_failures += record.failure_rate;
            max_fail = std::max(max_fail, record.failure_rate);
            min_fail = std::min(min_fail, record.failure_rate);
        }
        
        float avg_fail = sum_failures / recent_records.size();
        
        // Simple trend analysis: compare first and last 5 records
        int sample_size = std::min(5, static_cast<int>(recent_records.size()));
        float early_avg = 0, late_avg = 0;
        
        for (int i = 0; i < sample_size; ++i) {
            early_avg += recent_records[i].failure_rate;
            late_avg += recent_records[recent_records.size() - sample_size + i].failure_rate;
        }
        
        early_avg /= sample_size;
        late_avg /= sample_size;
        
        TrendAnalysis trend;
        trend.avg_failure_rate = avg_fail;
        trend.max_failure_rate = max_fail;
        trend.min_failure_rate = min_fail;
        trend.is_improving = late_avg < early_avg;
        trend.is_degrading = late_avg > early_avg;
        
        return trend;
    }
};
```

## Hands-on Exercises

### Exercise 1: Performance Profiling Pipeline
Create a complete profiling pipeline for CUTLASS operations.

```cpp
// TODO: Create a profiling pipeline that:
// 1. Profiles CUTLASS GEMM operations with Nsight Compute
// 2. Collects memory bandwidth and occupancy metrics
// 3. Generates performance reports
// 4. Identifies bottlenecks and suggests optimizations
```

### Exercise 2: Memory Optimization for Specific Shapes
Optimize memory access for specific matrix shapes commonly used in transformers.

```cpp
// TODO: Optimize for transformer-specific shapes like:
// 1. Self-attention: [seq_len, seq_len] matrices
// 2. Feed-forward: [hidden_size, 4*hidden_size] matrices
// 3. Batch processing with varying sequence lengths
// 4. Include padding optimization
```

### Exercise 3: Production Accuracy Validation
Implement a production-ready accuracy validation system.

```cpp
// TODO: Create an accuracy validation system that:
// 1. Validates CUTLASS results against reference implementations
// 2. Monitors accuracy trends over time
// 3. Provides alerts for accuracy degradation
// 4. Includes statistical analysis and reporting
```

## Solutions to Exercises

### Solution 1: Performance Profiling Pipeline
```cpp
#include <fstream>
#include <sstream>

class ComprehensiveProfiler {
public:
    struct ProfilingReport {
        std::string operation_name;
        int m, n, k;
        float execution_time_ms;
        float gflops;
        float bandwidth_gb_s;
        float occupancy_percent;
        std::string bottleneck_analysis;
        std::vector<std::string> optimization_suggestions;
    };
    
    static ProfilingReport profile_cutlass_operation(
        std::function<void()> operation,
        const char* name,
        int m, int n, int k) {
        
        ProfilingReport report;
        report.operation_name = name;
        report.m = m;
        report.n = n;
        report.k = k;
        
        // Time the operation
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        
        cudaEventRecord(start);
        operation();
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        cudaEventElapsedTime(&report.execution_time_ms, start, stop);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Calculate performance metrics
        float ops = 2.0f * m * n * k;  // multiply-add operations
        report.gflops = (ops / 1e9f) / (report.execution_time_ms / 1000.0f);
        
        // Estimate memory bandwidth
        size_t bytes = (m * k + k * n + m * n) * sizeof(float);
        report.bandwidth_gb_s = (bytes / 1e9f) / (report.execution_time_ms / 1000.0f);
        
        // Analyze potential bottlenecks
        report.bottleneck_analysis = analyze_bottlenecks(report);
        
        // Generate optimization suggestions
        report.optimization_suggestions = generate_optimizations(report);
        
        return report;
    }
    
    static void generate_profiling_report(const ProfilingReport& report) {
        std::ofstream file("performance_report.txt", std::ios::app);
        
        file << "=== Performance Report ===" << std::endl;
        file << "Operation: " << report.operation_name << std::endl;
        file << "Dimensions: " << report.m << "x" << report.n << "x" << report.k << std::endl;
        file << "Execution Time: " << report.execution_time_ms << " ms" << std::endl;
        file << "GFLOPS: " << report.gflops << std::endl;
        file << "Bandwidth: " << report.bandwidth_gb_s << " GB/s" << std::endl;
        file << "Bottleneck: " << report.bottleneck_analysis << std::endl;
        file << "Suggestions:" << std::endl;
        
        for (const auto& suggestion : report.optimization_suggestions) {
            file << "  - " << suggestion << std::endl;
        }
        
        file << std::endl;
        file.close();
    }

private:
    static std::string analyze_bottlenecks(const ProfilingReport& report) {
        if (report.gflops < 50) {
            return "Compute-bound: Low GFLOPS suggest compute limitations";
        } else if (report.bandwidth_gb_s < 500) {
            return "Memory-bound: Low bandwidth suggests memory access issues";
        } else {
            return "Balanced: Good balance between compute and memory";
        }
    }
    
    static std::vector<std::string> generate_optimizations(const ProfilingReport& report) {
        std::vector<std::string> suggestions;
        
        if (report.gflops < 50) {
            suggestions.push_back("Consider using Tensor Cores for higher compute throughput");
            suggestions.push_back("Try different threadblock shapes for better compute utilization");
        }
        
        if (report.bandwidth_gb_s < 500) {
            suggestions.push_back("Optimize memory access patterns for better coalescing");
            suggestions.push_back("Increase shared memory usage to reduce global memory traffic");
            suggestions.push_back("Consider data layout transformations");
        }
        
        if (report.execution_time_ms > 10.0f) {
            suggestions.push_back("Consider batching multiple operations together");
            suggestions.push_back("Use CUDA streams for overlapping computation and memory transfers");
        }
        
        return suggestions;
    }
};

// Usage example
void run_comprehensive_profiling() {
    // Example CUTLASS operation
    auto cutlass_operation = []() {
        // CUTLASS GEMM kernel launch
        // This would be your actual CUTLASS call
    };
    
    auto report = ComprehensiveProfiler::profile_cutlass_operation(
        cutlass_operation, "FP16_GEMM", 1024, 1024, 1024);
    
    ComprehensiveProfiler::generate_profiling_report(report);
}
```

### Solution 2: Memory Optimization for Specific Shapes
```cuda
// Optimized memory access for transformer-specific shapes
class TransformerMemoryOptimizer {
public:
    // Self-attention matrix optimization [seq_len, seq_len]
    template<typename Element>
    struct SelfAttentionMemory {
        static_assert(std::is_same_v<Element, cutlass::half_t> || 
                     std::is_same_v<Element, float>, 
                     "Only FP16 and FP32 supported");
        
        // Optimize for attention score matrix (typically [4096, 4096] or similar powers of 2)
        static constexpr int kTileSize = 64;  // Good for attention matrices
        
        CUTLASS_DEVICE
        static void load_tile_coalesced(
            Element const* global_ptr,
            Element* shared_ptr,
            int seq_len,
            int tile_row,
            int tile_col,
            int thread_row,
            int thread_col) {
            
            // Ensure coalesced access for attention computation
            int global_row = tile_row + thread_row;
            int global_col = tile_col + thread_col;
            
            if (global_row < seq_len && global_col < seq_len) {
                int global_idx = global_row * seq_len + global_col;
                int shared_idx = thread_row * kTileSize + thread_col;
                
                shared_ptr[shared_idx] = global_ptr[global_idx];
            }
        }
        
        // Padding for attention matrices to avoid bank conflicts
        static constexpr int get_padded_size(int original_size) {
            // Pad to multiple of 32 for better memory access
            return ((original_size + 31) / 32) * 32;
        }
    };
    
    // Feed-forward network optimization [hidden_size, 4*hidden_size]
    template<typename Element>
    struct FeedForwardMemory {
        static constexpr int kHiddenSize = 4096;  // Common transformer hidden size
        static constexpr int kIntermediateSize = 4 * kHiddenSize;
        
        CUTLASS_DEVICE
        static void optimize_for_ffn(
            Element const* input,
            Element* output,
            int batch_size,
            int seq_len) {
            
            // Process in tiles optimized for FFN dimensions
            const int tile_m = 128;  // Hidden dimension tile
            const int tile_n = 128;  // Intermediate dimension tile
            
            for (int b = 0; b < batch_size; ++b) {
                for (int s = 0; s < seq_len; ++s) {
                    for (int m = 0; m < kHiddenSize; m += tile_m) {
                        for (int n = 0; n < kIntermediateSize; n += tile_n) {
                            // Process tile [m:m+tile_m, n:n+tile_n]
                            process_ffn_tile(
                                input + (b * seq_len + s) * kHiddenSize + m,
                                output + (b * seq_len + s) * kIntermediateSize + n,
                                min(tile_m, kHiddenSize - m),
                                min(tile_n, kIntermediateSize - n),
                                kHiddenSize, kIntermediateSize);
                        }
                    }
                }
            }
        }
        
    private:
        CUTLASS_DEVICE
        static void process_ffn_tile(
            Element const* input_tile,
            Element* output_tile,
            int tile_m, int tile_n,
            int input_ld, int output_ld) {
            
            // Process the tile with optimized memory access
            for (int i = 0; i < tile_m; ++i) {
                for (int j = 0; j < tile_n; ++j) {
                    // Coalesced access pattern
                    output_tile[i * output_ld + j] = 
                        input_tile[i * input_ld] * Element(2.0f); // Example operation
                }
            }
        }
    };
    
    // Batch processing with variable sequence lengths
    struct VariableSequenceOptimizer {
        CUTLASS_DEVICE
        static void process_variable_sequences(
            cutlass::half_t* packed_data,  // Packed sequences with padding
            int const* sequence_lengths,   // Actual length of each sequence
            int max_seq_len,
            int batch_size,
            int hidden_size) {
            
            // Process sequences in a way that minimizes padding waste
            int total_tokens = 0;
            for (int i = 0; i < batch_size; ++i) {
                total_tokens += sequence_lengths[i];
            }
            
            // Use a token-based processing approach
            process_tokens_individually(packed_data, sequence_lengths, 
                                      batch_size, max_seq_len, hidden_size);
        }
        
    private:
        CUTLASS_DEVICE
        static void process_tokens_individually(
            cutlass::half_t* packed_data,
            int const* sequence_lengths,
            int batch_size,
            int max_seq_len,
            int hidden_size) {
            
            // Each thread block processes tokens from multiple sequences
            int token_id = blockIdx.x * blockDim.x + threadIdx.x;
            
            // Map linear token ID to sequence and position
            int current_token = 0;
            for (int seq = 0; seq < batch_size; ++seq) {
                if (token_id < current_token + sequence_lengths[seq]) {
                    int pos_in_seq = token_id - current_token;
                    int data_offset = seq * max_seq_len * hidden_size + pos_in_seq * hidden_size;
                    
                    // Process this token
                    process_single_token(packed_data + data_offset, hidden_size);
                    break;
                }
                current_token += sequence_lengths[seq];
            }
        }
        
        CUTLASS_DEVICE
        static void process_single_token(cutlass::half_t* token_data, int hidden_size) {
            // Process a single token's hidden representation
            for (int i = 0; i < hidden_size; ++i) {
                token_data[i] = cutlass::half_t(float(token_data[i]) * 1.1f); // Example
            }
        }
    };
};
```

### Solution 3: Production Accuracy Validation
```cpp
#include <thread>
#include <atomic>

class ProductionAccuracyValidation {
private:
    std::atomic<bool> validation_enabled_{true};
    std::atomic<float> tolerance_threshold_{1e-4f};
    std::atomic<int> validation_frequency_{100};  // Every 100 operations
    std::atomic<int> operation_counter_{0};
    
    // Reference implementation for validation
    template<typename T>
    static void reference_gemm(
        T const* A, T const* B, T* C,
        int m, int n, int k,
        T alpha, T beta) {
        
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                T sum = T(0);
                for (int l = 0; l < k; ++l) {
                    sum += A[i * k + l] * B[l * n + j];
                }
                C[i * n + j] = alpha * sum + beta * C[i * n + j];
            }
        }
    }

public:
    template<typename T>
    bool validate_cutlass_result(
        T const* cutlass_result,
        T const* input_A,
        T const* input_B,
        int m, int n, int k,
        T alpha = T(1), T beta = T(0)) {
        
        if (!validation_enabled_) {
            return true;  // Skip validation if disabled
        }
        
        // Only validate periodically based on frequency
        int count = operation_counter_.fetch_add(1);
        if (count % validation_frequency_ != 0) {
            return true;  // Skip this validation
        }
        
        // Create reference result
        std::vector<T> reference_result(m * n);
        reference_gemm(input_A, input_B, reference_result.data(), m, n, k, alpha, beta);
        
        // Validate accuracy
        auto stats = AccuracyValidator::compute_accuracy_stats(
            cutlass_result, reference_result.data(), m * n, 
            T(tolerance_threshold_));
        
        // Log validation results
        log_validation_result(stats, m, n, k, alpha, beta);
        
        // Trigger alert if accuracy is poor
        if (stats.failure_rate > 0.05f) {
            trigger_accuracy_alert(stats, m, n, k);
            return false;  // Fail validation
        }
        
        return true;  // Pass validation
    }
    
    void set_validation_parameters(float tolerance, int frequency) {
        tolerance_threshold_ = tolerance;
        validation_frequency_ = frequency;
    }
    
    void enable_validation(bool enabled) {
        validation_enabled_ = enabled;
    }
    
    // Background validation thread
    void start_background_validation() {
        std::thread([this]() {
            while (validation_enabled_) {
                // Periodically check for validation tasks
                std::this_thread::sleep_for(std::chrono::seconds(10));
                
                // Could implement continuous monitoring here
                check_system_health();
            }
        }).detach();
    }

private:
    void log_validation_result(
        const typename AccuracyValidator::AccuracyStats& stats,
        int m, int n, int k, float alpha, float beta) {
        
        std::ofstream log_file("accuracy_log.csv", std::ios::app);
        if (log_file.is_open()) {
            auto now = std::chrono::system_clock::now();
            auto time_t = std::chrono::system_clock::to_time_t(now);
            
            log_file << std::put_time(std::localtime(&time_t), "%Y-%m-%d %H:%M:%S") << ","
                     << m << "," << n << "," << k << ","
                     << alpha << "," << beta << ","
                     << stats.max_error << ","
                     << stats.mean_error << ","
                     << stats.failure_rate << ","
                     << (stats.failure_rate <= 0.05f ? "PASS" : "FAIL") << std::endl;
        }
    }
    
    void trigger_accuracy_alert(
        const typename AccuracyValidator::AccuracyStats& stats,
        int m, int n, int k) {
        
        std::cerr << "ACCURACY ALERT: Poor numerical accuracy detected!" << std::endl;
        std::cerr << "  Dimensions: " << m << "x" << n << "x" << k << std::endl;
        std::cerr << "  Max error: " << stats.max_error << std::endl;
        std::cerr << "  Failure rate: " << (stats.failure_rate * 100) << "%" << std::endl;
        std::cerr << "  Mean error: " << stats.mean_error << std::endl;
        
        // In production, this might trigger monitoring systems
        // send_alert_to_monitoring_system();
    }
    
    void check_system_health() {
        // Check GPU temperature, utilization, etc.
        // This is a placeholder for actual health checks
    }
};

// Global validation instance for production use
ProductionAccuracyValidation g_accuracy_validator;

// Macro for easy validation in production code
#define VALIDATE_CUTLASS_RESULT(cutlass_res, a, b, m, n, k, alpha, beta) \
    g_accuracy_validator.validate_cutlass_result( \
        cutlass_res, a, b, m, n, k, alpha, beta)
```

## Advanced Topic: Performance Regression Testing

### Automated Performance Testing
```cpp
// Performance regression testing framework
class PerformanceRegressionTester {
private:
    struct BaselinePerformance {
        float gflops;
        float bandwidth_gb_s;
        float execution_time_ms;
        std::chrono::system_clock::time_point timestamp;
    };
    
    std::map<std::string, BaselinePerformance> baseline_results_;
    
public:
    void establish_baseline(
        const std::string& test_name,
        float gflops, float bandwidth, float time_ms) {
        
        BaselinePerformance baseline;
        baseline.gflops = gflops;
        baseline.bandwidth_gb_s = bandwidth;
        baseline.execution_time_ms = time_ms;
        baseline.timestamp = std::chrono::system_clock::now();
        
        baseline_results_[test_name] = baseline;
    }
    
    enum class PerformanceChange {
        REGRESSION,   // Performance decreased
        IMPROVEMENT,  // Performance increased
        SAME        // No significant change
    };
    
    PerformanceChange check_regression(
        const std::string& test_name,
        float current_gflops,
        float current_bandwidth,
        float current_time) {
        
        auto it = baseline_results_.find(test_name);
        if (it == baseline_results_.end()) {
            // No baseline, can't compare
            establish_baseline(test_name, current_gflops, current_bandwidth, current_time);
            return PerformanceChange::SAME;
        }
        
        const auto& baseline = it->second;
        
        // Check for regressions (>10% decrease in performance)
        if (current_gflops < baseline.gflops * 0.9f) {
            return PerformanceChange::REGRESSION;
        }
        
        // Check for improvements (>10% increase in performance)
        if (current_gflops > baseline.gflops * 1.1f) {
            return PerformanceChange::IMPROVEMENT;
        }
        
        return PerformanceChange::SAME;
    }
    
    void run_regression_suite() {
        std::vector<std::tuple<std::string, int, int, int>> test_cases = {
            {"small_gemm", 512, 512, 512},
            {"medium_gemm", 1024, 1024, 1024},
            {"large_gemm", 2048, 2048, 2048},
            {"rectangular_gemm", 4096, 1024, 1024}
        };
        
        for (const auto& [name, m, n, k] : test_cases) {
            float gflops, bandwidth, time_ms;
            
            // Run the test case
            run_performance_test(m, n, k, gflops, bandwidth, time_ms);
            
            // Check for regression
            auto change = check_regression(name, gflops, bandwidth, time_ms);
            
            switch (change) {
                case PerformanceChange::REGRESSION:
                    std::cout << "PERFORMANCE REGRESSION in " << name << std::endl;
                    break;
                case PerformanceChange::IMPROVEMENT:
                    std::cout << "PERFORMANCE IMPROVEMENT in " << name << std::endl;
                    break;
                case PerformanceChange::SAME:
                    std::cout << "PERFORMANCE STABLE in " << name << std::endl;
                    break;
            }
        }
    }

private:
    void run_performance_test(int m, int n, int k, float& gflops, float& bandwidth, float& time_ms) {
        // Implementation would run actual performance test
        // This is a placeholder
        time_ms = 10.0f;  // Example
        float ops = 2.0f * m * n * k;
        gflops = (ops / 1e9f) / (time_ms / 1000.0f);
        size_t bytes = (m * k + k * n + m * n) * sizeof(float);
        bandwidth = (bytes / 1e9f) / (time_ms / 1000.0f);
    }
};
```

## Quiz Questions

1. What are the key metrics to monitor when profiling CUTLASS performance?

2. How does memory coalescing affect GPU performance and how can it be optimized?

3. What factors limit occupancy in CUDA kernels and how can they be addressed?

4. How do you measure and validate numerical accuracy in production systems?

5. What is the significance of bandwidth utilization vs compute utilization in performance optimization?

## Summary
Module 10 covered comprehensive performance optimization and profiling techniques for CUTLASS, including GPU profiling tools, memory bandwidth optimization, occupancy maximization, asynchronous operations, and numerical accuracy validation. These advanced techniques are essential for deploying CUTLASS in production environments where performance and reliability are critical.