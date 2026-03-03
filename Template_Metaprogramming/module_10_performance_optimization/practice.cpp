#include <iostream>
#include <vector>
#include <type_traits>
#include <chrono>
#include <cmath>
#include <algorithm>
#include <functional>
#include <random>

// Module 10: Performance Optimization and Profiling Practice
// Hands-on tutorial for advanced performance analysis and optimization

/*
 * EXERCISE 1: GPU PROFILING TOOLS
 * Using profiling tools to analyze CUTLASS performance
 */
class NsightProfileSimulator {
public:
    struct ProfileMetrics {
        float achieved_occupancy;
        float sm_efficiency;
        float dram_read_throughput;
        float dram_write_throughput;
        float tensor_core_utilization;
    };

    static void simulate_profile_kernel_launch(
        std::function<void()> kernel_func,
        const char* kernel_name,
        ProfileMetrics& metrics) {

        std::cout << "Simulating profiling for kernel: " << kernel_name << std::endl;

        // Simulate kernel execution
        auto start = std::chrono::high_resolution_clock::now();
        kernel_func();
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Kernel execution time: " << duration.count() << " microseconds" << std::endl;

        // Simulated metrics collection
        metrics.achieved_occupancy = 0.85f;  // 85% occupancy
        metrics.sm_efficiency = 0.92f;      // 92% SM efficiency
        metrics.dram_read_throughput = 850.0f;  // GB/s
        metrics.dram_write_throughput = 420.0f; // GB/s
        metrics.tensor_core_utilization = 0.78f; // 78% utilization

        std::cout << "Profiling completed for: " << kernel_name << std::endl;
    }

    // Profile memory access patterns
    static void simulate_profile_memory_access(
        void* ptr, size_t size,
        const char* description) {

        std::cout << "Memory profiling for " << description << ":" << std::endl;
        std::cout << "  Address: " << ptr << std::endl;
        std::cout << "  Size: " << size << " bytes" << std::endl;
        std::cout << "  Alignment: " <<
            (reinterpret_cast<uintptr_t>(ptr) % 128) << " bytes offset from 128-byte boundary" << std::endl;

        // Check for coalescing opportunities
        size_t addr = reinterpret_cast<uintptr_t>(ptr);
        if (addr % 128 == 0) {
            std::cout << "  ✓ Aligned to 128-byte boundary" << std::endl;
        } else {
            std::cout << "  ⚠ Not aligned to 128-byte boundary" << std::endl;
        }
    }
};

void exercise_gpu_profiling_tools() {
    std::cout << "\n=== Exercise 1: GPU Profiling Tools ===" << std::endl;

    std::cout << "GPU profiling is essential for identifying performance bottlenecks." << std::endl;
    std::cout << "Common tools include Nsight Compute, nvprof, and custom profiling utilities." << std::endl;

    NsightProfileSimulator::ProfileMetrics metrics;

    // Simulate a kernel function
    auto kernel_func = []() {
        // Simulate some computation
        volatile float sum = 0.0f;
        for (int i = 0; i < 1000000; ++i) {
            sum += sinf(i * 0.001f);
        }
    };

    NsightProfileSimulator::simulate_profile_kernel_launch(
        kernel_func, "simulated_cutlass_gemm", metrics);

    std::cout << "\nSimulated Profiling Results:" << std::endl;
    std::cout << "  Achieved Occupancy: " << (metrics.achieved_occupancy * 100) << "%" << std::endl;
    std::cout << "  SM Efficiency: " << (metrics.sm_efficiency * 100) << "%" << std::endl;
    std::cout << "  DRAM Read Throughput: " << metrics.dram_read_throughput << " GB/s" << std::endl;
    std::cout << "  DRAM Write Throughput: " << metrics.dram_write_throughput << " GB/s" << std::endl;
    std::cout << "  Tensor Core Utilization: " << (metrics.tensor_core_utilization * 100) << "%" << std::endl;

    // Simulate memory profiling
    std::vector<float> data(1024);
    NsightProfileSimulator::simulate_profile_memory_access(
        data.data(), data.size() * sizeof(float), "input_buffer");

    std::cout << "\nProfiling best practices:" << std::endl;
    std::cout << "1. Profile at different problem sizes" << std::endl;
    std::cout << "2. Monitor multiple metrics simultaneously" << std::endl;
    std::cout << "3. Compare against theoretical peaks" << std::endl;
    std::cout << "4. Identify bottlenecks (compute vs memory bound)" << std::endl;
}

/*
 * EXERCISE 2: MEMORY BANDWIDTH OPTIMIZATION
 * Optimizing memory access patterns and bandwidth utilization
 */
class MemoryBandwidthOptimizer {
public:
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

        // Calculate theoretical memory bandwidth (simulated)
        float peak_bandwidth_gb_s = 900.0f; // Simulated peak for modern GPU

        // Calculate required memory operations
        size_t bytes_read = (problem_size_m * problem_size_k +
                           problem_size_k * problem_size_n) * data_type_size_bytes;
        size_t bytes_written = problem_size_m * problem_size_n * data_type_size_bytes;
        size_t total_bytes = bytes_read + bytes_written;

        // Estimate computation time based on peak TFLOPS
        float peak_tflops = 100.0f; // Simulated peak for modern GPU
        float ops = 2.0f * problem_size_m * problem_size_n * problem_size_k;  // 2 ops per multiply-add
        float estimated_time_s = (ops / 1e12f) / peak_tflops;  // Time at peak performance

        analysis.theoretical_bandwidth_gb_s = (total_bytes / 1e9f) / estimated_time_s;
        analysis.achieved_bandwidth_gb_s = 750.0f; // Simulated achieved bandwidth
        analysis.bandwidth_utilization_percent =
            (analysis.achieved_bandwidth_gb_s / peak_bandwidth_gb_s) * 100.0f;

        // Check for coalescing (simplified check)
        analysis.has_coalesced_access = (problem_size_n % 32 == 0); // Multiple of warp size

        // Bank conflict estimation
        analysis.bank_conflicts_per_access = estimate_bank_conflicts(problem_size_m, problem_size_n);

        return analysis;
    }

    // Optimize memory access for better bandwidth
    template<typename Element>
    static void optimize_memory_layout(
        Element* input, Element* output,
        int rows, int cols) {

        // Process in tiles to improve cache locality
        const int TILE_SIZE = 64;

        for (int i = 0; i < rows; i += TILE_SIZE) {
            for (int j = 0; j < cols; j += TILE_SIZE) {
                // Process tile [i:i+TILE_SIZE, j:j+TILE_SIZE]
                for (int ii = i; ii < std::min(i + TILE_SIZE, rows); ++ii) {
                    for (int jj = j; jj < std::min(j + TILE_SIZE, cols); ++jj) {
                        // Access is coalesced when processing by columns in column-major
                        output[ii * cols + jj] = input[ii * cols + jj];
                    }
                }
            }
        }
    }

    static void demonstrate_memory_optimization() {
        std::cout << "\nMemory Bandwidth Optimization:" << std::endl;

        // Example analysis
        auto analysis = analyze_memory_access(1024, 1024, 512, sizeof(float));

        std::cout << "Problem size: 1024x1024x512" << std::endl;
        std::cout << "Theoretical bandwidth: " << analysis.theoretical_bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "Achieved bandwidth: " << analysis.achieved_bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "Utilization: " << analysis.bandwidth_utilization_percent << "%" << std::endl;
        std::cout << "Coalesced access: " << (analysis.has_coalesced_access ? "Yes" : "No") << std::endl;
        std::cout << "Bank conflicts per access: " << analysis.bank_conflicts_per_access << std::endl;

        // Demonstrate tiled access
        std::vector<float> input(1024 * 1024);
        std::vector<float> output(1024 * 1024);

        auto start = std::chrono::high_resolution_clock::now();
        optimize_memory_layout(input.data(), output.data(), 1024, 1024);
        auto end = std::chrono::high_resolution_clock::now();

        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        std::cout << "Tiled memory access took: " << duration.count() << " microseconds" << std::endl;
    }

private:
    static int estimate_bank_conflicts(int rows, int cols) {
        // Simplified bank conflict estimation
        if (cols % 32 == 0) {
            return 32; // Worst case: all threads access same bank
        }
        return 1; // Best case: no conflicts
    }
};

void exercise_memory_bandwidth_optimization() {
    std::cout << "\n=== Exercise 2: Memory Bandwidth Optimization ===" << std::endl;

    std::cout << "Memory bandwidth optimization is crucial for achieving peak performance." << std::endl;
    std::cout << "Key areas include coalesced access, bank conflict avoidance, and cache optimization." << std::endl;

    MemoryBandwidthOptimizer::demonstrate_memory_optimization();

    std::cout << "\nMemory optimization techniques:" << std::endl;
    std::cout << "1. Coalesced memory access patterns" << std::endl;
    std::cout << "2. Shared memory bank conflict avoidance" << std::endl;
    std::cout << "3. Memory access coalescing" << std::endl;
    std::cout << "4. Cache-friendly data layouts" << std::endl;
    std::cout << "5. Memory prefetching" << std::endl;
}

/*
 * EXERCISE 3: OCCUPANCY OPTIMIZATION
 * Maximizing occupancy for different problem sizes
 */
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
        int block_size_x, int block_size_y, int block_size_z,
        size_t dynamic_smem_bytes) {

        OccupancyInfo info{};

        // Simulated GPU properties
        const int max_threads_per_sm = 2048;
        const int max_blocks_per_sm = 32;
        const size_t shared_mem_per_sm = 164000; // 164KB
        const int regs_per_sm = 65536;

        int block_size = block_size_x * block_size_y * block_size_z;

        // Calculate limiting factors
        int max_blocks_by_threads = max_threads_per_sm / block_size;
        int max_blocks_by_blocks = max_blocks_per_sm;
        int max_blocks_by_smem = (dynamic_smem_bytes > 0) ?
                                 shared_mem_per_sm / dynamic_smem_bytes : max_blocks_per_sm;

        info.max_active_blocks_per_sm = std::min({max_blocks_by_threads,
                                                 max_blocks_by_blocks,
                                                 max_blocks_by_smem});

        info.theoretical_occupancy_percent =
            (info.max_active_blocks_per_sm * block_size * 100) / max_threads_per_sm;

        // Determine limiting resource
        if (max_blocks_by_smem <= max_blocks_by_threads &&
            max_blocks_by_smem <= max_blocks_by_blocks) {
            info.limiting_resource = 1; // Shared memory
        } else if (max_blocks_by_threads <= max_blocks_by_smem &&
                   max_blocks_by_threads <= max_blocks_by_blocks) {
            info.limiting_resource = 2; // Registers
        } else {
            info.limiting_resource = 0; // Blocks
        }

        info.shared_memory_per_block = dynamic_smem_bytes;
        info.registers_per_thread = 32; // Simulated value

        return info;
    }

    // Optimize block size for maximum occupancy
    static int find_optimal_block_size(size_t dynamic_smem_bytes) {
        // Simulated occupancy calculation
        // In practice, this would use CUDA occupancy calculator

        // Try different block sizes and return the one with highest occupancy
        std::vector<int> block_sizes = {64, 128, 256, 512};
        int best_size = 256; // Default
        int best_occupancy = 0;

        for (int size : block_sizes) {
            int blocks_per_sm = std::min(2048 / size, 32); // Threads and blocks limits
            if (dynamic_smem_bytes > 0) {
                blocks_per_sm = std::min(blocks_per_sm, static_cast<int>(164000 / dynamic_smem_bytes));
            }

            int occupancy = (blocks_per_sm * size * 100) / 2048;
            if (occupancy > best_occupancy) {
                best_occupancy = occupancy;
                best_size = size;
            }
        }

        return best_size;
    }

    static void demonstrate_occupancy_analysis() {
        std::cout << "\nOccupancy Analysis:" << std::endl;

        // Example 1: Small block size
        auto info1 = analyze_occupancy(64, 1, 1, 0);
        std::cout << "Block size 64x1x1, no shared memory:" << std::endl;
        std::cout << "  Max active blocks per SM: " << info1.max_active_blocks_per_sm << std::endl;
        std::cout << "  Theoretical occupancy: " << info1.theoretical_occupancy_percent << "%" << std::endl;
        std::cout << "  Limiting resource: " <<
            (info1.limiting_resource == 0 ? "Blocks" :
             info1.limiting_resource == 1 ? "Shared Memory" : "Registers") << std::endl;

        // Example 2: Large block size with shared memory
        auto info2 = analyze_occupancy(256, 1, 1, 32768); // 32KB shared memory
        std::cout << "\nBlock size 256x1x1, 32KB shared memory:" << std::endl;
        std::cout << "  Max active blocks per SM: " << info2.max_active_blocks_per_sm << std::endl;
        std::cout << "  Theoretical occupancy: " << info2.theoretical_occupancy_percent << "%" << std::endl;
        std::cout << "  Limiting resource: " <<
            (info2.limiting_resource == 0 ? "Blocks" :
             info2.limiting_resource == 1 ? "Shared Memory" : "Registers") << std::endl;

        // Find optimal block size
        int optimal_size = find_optimal_block_size(16384); // 16KB shared memory
        std::cout << "\nOptimal block size for 16KB shared memory: " << optimal_size << std::endl;
    }
};

void exercise_occupancy_optimization() {
    std::cout << "\n=== Exercise 3: Occupancy Optimization ===" << std::endl;

    std::cout << "Occupancy optimization maximizes the utilization of GPU streaming multiprocessors." << std::endl;
    std::cout << "Higher occupancy can hide memory latency and improve performance." << std::endl;

    OccupancyOptimizer::demonstrate_occupancy_analysis();

    std::cout << "\nOccupancy optimization strategies:" << std::endl;
    std::cout << "1. Balance block size to maximize occupancy" << std::endl;
    std::cout << "2. Minimize shared memory usage per block" << std::endl;
    std::cout << "3. Reduce register usage per thread" << std::endl;
    std::cout << "4. Use occupancy calculator APIs" << std::endl;
    std::cout << "5. Consider trade-offs between occupancy and resource usage" << std::endl;
}

/*
 * EXERCISE 4: ASYNCHRONOUS OPERATIONS
 * Implementing asynchronous operations for better performance
 */
class AsyncOperationsSimulator {
public:
    struct StreamInfo {
        int id;
        std::chrono::microseconds execution_time;
        bool is_active;
    };

    static void simulate_async_gemm_batch(
        int num_batches, int batch_size) {

        std::cout << "\nSimulating asynchronous GEMM operations:" << std::endl;
        std::cout << "Number of batches: " << num_batches << std::endl;
        std::cout << "Batch size: " << batch_size << std::endl;

        // Simulate operations on multiple streams
        std::vector<StreamInfo> streams(4); // Simulate 4 streams

        for (int i = 0; i < 4; ++i) {
            streams[i].id = i;
            streams[i].is_active = false;
        }

        // Simulate overlapping computation
        auto start_time = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < num_batches; ++batch) {
            int stream_idx = batch % 4;

            // Simulate memory copy
            std::this_thread::sleep_for(std::chrono::microseconds(10)); // Simulate memory transfer

            // Simulate computation
            volatile float sum = 0.0f;
            for (int i = 0; i < batch_size * 100; ++i) {
                sum += sinf(i * 0.001f);
            }

            // Record completion time
            streams[stream_idx].is_active = true;
            streams[stream_idx].execution_time =
                std::chrono::duration_cast<std::chrono::microseconds>(
                    std::chrono::high_resolution_clock::now() - start_time);
        }

        auto total_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - start_time);

        std::cout << "Total execution time with async operations: " << total_time.count() << " microseconds" << std::endl;

        // Compare with synchronous execution
        auto sync_start = std::chrono::high_resolution_clock::now();
        for (int batch = 0; batch < num_batches; ++batch) {
            volatile float sum = 0.0f;
            for (int i = 0; i < batch_size * 100; ++i) {
                sum += sinf(i * 0.001f);
            }
        }
        auto sync_time = std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::high_resolution_clock::now() - sync_start);

        std::cout << "Synchronous execution time: " << sync_time.count() << " microseconds" << std::endl;
        std::cout << "Speedup from async operations: " <<
            (float)sync_time.count() / total_time.count() << "x" << std::endl;
    }

    static void demonstrate_stream_synchronization() {
        std::cout << "\nStream synchronization example:" << std::endl;

        // Simulate dependency management
        std::cout << "Step 1: Launch computation on Stream 0" << std::endl;
        std::this_thread::sleep_for(std::chrono::microseconds(50));

        std::cout << "Step 2: Wait for Stream 0 to complete before starting Stream 1" << std::endl;
        std::this_thread::sleep_for(std::chrono::microseconds(10));

        std::cout << "Step 3: Launch dependent computation on Stream 1" << std::endl;
        std::this_thread::sleep_for(std::chrono::microseconds(40));

        std::cout << "Step 4: Wait for all streams to complete" << std::endl;
        std::cout << "All operations completed successfully" << std::endl;
    }
};

void exercise_asynchronous_operations() {
    std::cout << "\n=== Exercise 4: Asynchronous Operations ===" << std::endl;

    std::cout << "Asynchronous operations overlap computation with memory transfers." << std::endl;
    std::cout << "This can significantly improve performance by hiding memory latency." << std::endl;

    AsyncOperationsSimulator::simulate_async_gemm_batch(8, 1000);
    AsyncOperationsSimulator::demonstrate_stream_synchronization();

    std::cout << "\nAsynchronous operation benefits:" << std::endl;
    std::cout << "1. Overlapping computation with memory transfers" << std::endl;
    std::cout << "2. Better GPU utilization" << std::endl;
    std::cout << "3. Reduced overall execution time" << std::endl;
    std::cout << "4. Proper synchronization to maintain correctness" << std::endl;
}

/*
 * EXERCISE 5: NUMERICAL ACCURACY IN PRODUCTION
 * Analyzing and maintaining numerical accuracy
 */
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
                float rel_error = std::abs((ref_val - comp_val) / ref_val);
                max_error = std::max(max_error, rel_error);

                if (rel_error > tolerance) {
                    error_count++;
                }
            } else {
                float abs_error = std::abs(comp_val);
                max_error = std::max(max_error, abs_error);

                if (abs_error > tolerance) {
                    error_count++;
                }
            }
        }

        if (error_count > 0) {
            std::cout << "Warning: " << error_count << "/" << size
                      << " values exceed tolerance of " << tolerance << std::endl;
        }

        return max_error;
    }

    // Statistical accuracy analysis
    struct AccuracyStats {
        float max_error;
        float mean_error;
        float rms_error;
        float variance;
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
        double sum_errors = 0.0;
        double sum_squared_errors = 0.0;
        int failed_count = 0;

        for (int i = 0; i < size; ++i) {
            T ref_val = reference[i];
            T comp_val = computed[i];

            float error;
            if (ref_val != T(0)) {
                error = std::abs((ref_val - comp_val) / ref_val);
            } else {
                error = std::abs(comp_val);
            }

            sum_errors += error;
            sum_squared_errors += error * error;

            if (error > tolerance) {
                failed_count++;
            }
        }

        stats.max_error = 0.0f;
        stats.mean_error = static_cast<float>(sum_errors / size);
        stats.rms_error = std::sqrt(static_cast<float>(sum_squared_errors / size));
        stats.total_comparisons = size;
        stats.failed_comparisons = failed_count;
        stats.failure_rate = static_cast<float>(failed_count) / size;

        // Calculate max error separately
        for (int i = 0; i < size; ++i) {
            T ref_val = reference[i];
            T comp_val = computed[i];

            float error;
            if (ref_val != T(0)) {
                error = std::abs((ref_val - comp_val) / ref_val);
            } else {
                error = std::abs(comp_val);
            }

            stats.max_error = std::max(stats.max_error, error);
        }

        return stats;
    }

    static void demonstrate_accuracy_validation() {
        std::cout << "\nAccuracy Validation Example:" << std::endl;

        const int size = 1000;
        std::vector<float> reference(size);
        std::vector<float> computed(size);

        // Generate reference data
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dis(0.1f, 10.0f);

        for (int i = 0; i < size; ++i) {
            reference[i] = dis(gen);
            // Simulate computed result with small error
            computed[i] = reference[i] * 1.001f + 0.0001f;
        }

        // Calculate accuracy metrics
        auto stats = compute_accuracy_stats(computed.data(), reference.data(), size, 0.01f);

        std::cout << "Accuracy Statistics:" << std::endl;
        std::cout << "  Max Error: " << stats.max_error << std::endl;
        std::cout << "  Mean Error: " << stats.mean_error << std::endl;
        std::cout << "  RMS Error: " << stats.rms_error << std::endl;
        std::cout << "  Total Comparisons: " << stats.total_comparisons << std::endl;
        std::cout << "  Failed Comparisons: " << stats.failed_comparisons << std::endl;
        std::cout << "  Failure Rate: " << (stats.failure_rate * 100) << "%" << std::endl;

        // Calculate max relative error
        float max_rel_error = calculate_max_relative_error(
            computed.data(), reference.data(), size, 0.01f);
        std::cout << "  Max Relative Error: " << max_rel_error << std::endl;
    }
};

void exercise_numerical_accuracy() {
    std::cout << "\n=== Exercise 5: Numerical Accuracy in Production ===" << std::endl;

    std::cout << "Numerical accuracy is critical in production environments." << std::endl;
    std::cout << "Techniques include error analysis, validation against reference implementations, and statistical analysis." << std::endl;

    AccuracyValidator::demonstrate_accuracy_validation();

    std::cout << "\nAccuracy validation techniques:" << std::endl;
    std::cout << "1. Relative error analysis" << std::endl;
    std::cout << "2. Absolute error analysis" << std::endl;
    std::cout << "3. Statistical error metrics (mean, variance, RMS)" << std::endl;
    std::cout << "4. ULP (Units in Last Place) error measurement" << std::endl;
    std::cout << "5. Validation against high-precision reference implementations" << std::endl;
}

/*
 * HANDS-ON CHALLENGES
 * Try implementing these performance optimization techniques
 */

// Challenge 1: Performance Profiler
class PerformanceProfiler {
public:
    struct PerformanceMetrics {
        float gflops;
        float bandwidth_gb_s;
        float occupancy_percent;
        float efficiency_percent;
        float kernel_time_ms;
        float memory_time_ms;
    };

    static PerformanceMetrics measure_performance(
        int m, int n, int k, float kernel_time_ms) {

        PerformanceMetrics metrics;

        // Calculate GFLOPS (2 operations per multiply-add)
        long long total_ops = 2LL * m * n * k;
        metrics.gflops = (total_ops / 1e9) / (kernel_time_ms / 1000.0f);

        // Calculate memory bandwidth (simplified)
        long long bytes_read = (m * k + k * n) * sizeof(float);
        long long bytes_written = m * n * sizeof(float);
        long long total_bytes = bytes_read + bytes_written;
        metrics.bandwidth_gb_s = (total_bytes / 1e9) / (kernel_time_ms / 1000.0f);

        // Simulated occupancy and efficiency
        metrics.occupancy_percent = 85.0f; // Example value
        metrics.efficiency_percent = 78.0f; // Example value
        metrics.kernel_time_ms = kernel_time_ms;
        metrics.memory_time_ms = kernel_time_ms * 0.1f; // 10% memory overhead

        return metrics;
    }

    static void print_performance_report(const PerformanceMetrics& metrics) {
        std::cout << "\nPerformance Report:" << std::endl;
        std::cout << "  Kernel Time: " << metrics.kernel_time_ms << " ms" << std::endl;
        std::cout << "  Memory Time: " << metrics.memory_time_ms << " ms" << std::endl;
        std::cout << "  Total Time: " << (metrics.kernel_time_ms + metrics.memory_time_ms) << " ms" << std::endl;
        std::cout << "  GFLOPS: " << metrics.gflops << std::endl;
        std::cout << "  Bandwidth: " << metrics.bandwidth_gb_s << " GB/s" << std::endl;
        std::cout << "  Occupancy: " << metrics.occupancy_percent << "%" << std::endl;
        std::cout << "  Efficiency: " << metrics.efficiency_percent << "%" << std::endl;
    }
};

// Challenge 2: Memory Access Optimizer
class MemoryAccessOptimizer {
public:
    enum AccessPattern {
        COALESCED,
        STRIDED,
        RANDOM
    };

    static AccessPattern analyze_access_pattern(int stride, int warp_size = 32) {
        if (stride == 1) {
            return COALESCED;
        } else if (stride == warp_size) {
            return STRIDED;
        } else {
            return RANDOM;
        }
    }

    static void suggest_optimizations(AccessPattern pattern) {
        switch(pattern) {
            case COALESCED:
                std::cout << "Current pattern is already optimal for coalescing" << std::endl;
                break;
            case STRIDED:
                std::cout << "Strided access pattern detected - consider transposing data layout" << std::endl;
                break;
            case RANDOM:
                std::cout << "Random access pattern detected - consider data reorganization or caching strategies" << std::endl;
                break;
        }
    }

    static void demonstrate_analysis() {
        std::cout << "\nMemory Access Pattern Analysis:" << std::endl;

        // Example patterns
        auto pattern1 = analyze_access_pattern(1); // Coalesced
        std::cout << "Stride 1: ";
        suggest_optimizations(pattern1);

        auto pattern2 = analyze_access_pattern(32); // Strided
        std::cout << "Stride 32: ";
        suggest_optimizations(pattern2);

        auto pattern3 = analyze_access_pattern(7); // Random
        std::cout << "Stride 7: ";
        suggest_optimizations(pattern3);
    }
};

// Challenge 3: Optimization Advisor
class OptimizationAdvisor {
public:
    struct BottleneckAnalysis {
        std::string bottleneck_type;  // "memory", "compute", "occupancy"
        float severity;  // 0.0 to 1.0
        std::vector<std::string> recommendations;
    };

    static BottleneckAnalysis analyze_bottleneck(
        float bandwidth_utilization,
        float compute_utilization,
        float occupancy) {

        BottleneckAnalysis analysis;
        analysis.severity = 0.0f;

        if (bandwidth_utilization < 0.6f) {
            analysis.bottleneck_type = "memory";
            analysis.severity = (0.6f - bandwidth_utilization) / 0.6f;
            analysis.recommendations = {
                "Optimize memory access patterns",
                "Reduce memory transactions",
                "Use memory prefetching",
                "Consider data compression"
            };
        } else if (compute_utilization < 0.7f) {
            analysis.bottleneck_type = "compute";
            analysis.severity = (0.7f - compute_utilization) / 0.7f;
            analysis.recommendations = {
                "Increase arithmetic intensity",
                "Optimize algorithms",
                "Use specialized instructions",
                "Consider kernel fusion"
            };
        } else if (occupancy < 0.7f) {
            analysis.bottleneck_type = "occupancy";
            analysis.severity = (0.7f - occupancy) / 0.7f;
            analysis.recommendations = {
                "Adjust block size",
                "Reduce shared memory usage",
                "Minimize register usage",
                "Balance resources"
            };
        } else {
            analysis.bottleneck_type = "none";
            analysis.severity = 0.0f;
            analysis.recommendations = {"System is well-balanced"};
        }

        return analysis;
    }

    static void print_advice(const BottleneckAnalysis& analysis) {
        std::cout << "\nBottleneck Analysis:" << std::endl;
        std::cout << "  Type: " << analysis.bottleneck_type << std::endl;
        std::cout << "  Severity: " << (analysis.severity * 100) << "%" << std::endl;
        std::cout << "  Recommendations:" << std::endl;
        for (const auto& rec : analysis.recommendations) {
            std::cout << "    - " << rec << std::endl;
        }
    }
};

void run_challenges() {
    std::cout << "\n=== Challenge Solutions ===" << std::endl;

    // Challenge 1: Performance Profiler
    std::cout << "\nChallenge 1 - Performance Profiler:" << std::endl;
    auto metrics = PerformanceProfiler::measure_performance(1024, 1024, 512, 2.5f);
    PerformanceProfiler::print_performance_report(metrics);

    // Challenge 2: Memory Access Optimizer
    std::cout << "\nChallenge 2 - Memory Access Optimizer:" << std::endl;
    MemoryAccessOptimizer::demonstrate_analysis();

    // Challenge 3: Optimization Advisor
    std::cout << "\nChallenge 3 - Optimization Advisor:" << std::endl;
    auto bottleneck = OptimizationAdvisor::analyze_bottleneck(0.5f, 0.8f, 0.6f);
    OptimizationAdvisor::print_advice(bottleneck);
}

int main() {
    std::cout << "Module 10: Performance Optimization and Profiling Practice - Hands-on Tutorial" << std::endl;

    // Run all exercises
    exercise_gpu_profiling_tools();
    exercise_memory_bandwidth_optimization();
    exercise_occupancy_optimization();
    exercise_asynchronous_operations();
    exercise_numerical_accuracy();

    // Run challenges
    run_challenges();

    std::cout << "\nSummary:" << std::endl;
    std::cout << "This module covered advanced performance optimization techniques including:" << std::endl;
    std::cout << "- GPU profiling tools and methodologies" << std::endl;
    std::cout << "- Memory bandwidth optimization strategies" << std::endl;
    std::cout << "- Occupancy optimization for maximum GPU utilization" << std::endl;
    std::cout << "- Asynchronous operations for improved performance" << std::endl;
    std::cout << "- Numerical accuracy validation in production" << std::endl;
    std::cout << "These skills are essential for achieving peak performance in real-world applications." << std::endl;

    return 0;
}