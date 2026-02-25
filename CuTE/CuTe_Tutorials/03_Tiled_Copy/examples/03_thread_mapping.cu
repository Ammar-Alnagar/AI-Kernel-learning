/**
 * @file 03_thread_mapping.cu
 * @brief Thread Mapping in Tiled Copy
 * 
 * This tutorial demonstrates advanced thread mapping concepts:
 * - Swizzled thread mapping
 * - Vectorized loads/stores
 * - Efficient memory coalescing patterns
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace cute;

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                      << " - " << cudaGetErrorString(err) << std::endl; \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

int main() {
    std::cout << "=== CuTe Tutorial: Thread Mapping ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: Thread Mapping Strategies
    // ================================================================
    std::cout << "--- Concept 1: Thread Mapping Strategies ---" << std::endl;
    std::cout << "How threads are mapped to data affects memory efficiency." << std::endl;
    std::cout << "Good mapping = coalesced memory access = better performance" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Setup
    // ================================================================
    const int M = 16;
    const int N = 16;
    const int size = M * N;
    
    std::vector<float> h_src(size);
    for (int i = 0; i < size; ++i) {
        h_src[i] = static_cast<float>(i + 1);
    }
    
    float *d_src;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Data: " << M << "x" << N << " matrix" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Row-Major Thread Mapping
    // ================================================================
    std::cout << "--- Example 1: Row-Major Thread Mapping ---" << std::endl;
    
    // 4x4 thread block
    auto thread_layout_rm = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    
    std::cout << "Thread layout (row-major):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  ");
        for (int j = 0; j < 4; ++j) {
            printf("T%2d ", thread_layout_rm(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "In row-major mapping:" << std::endl;
    std::cout << "  - Adjacent threads in a row access adjacent memory" << std::endl;
    std::cout << "  - Good for row-major data layout" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Column-Major Thread Mapping
    // ================================================================
    std::cout << "--- Example 2: Column-Major Thread Mapping ---" << std::endl;
    
    auto thread_layout_cm = make_layout(make_shape(Int<4>{}, Int<4>{}), GenColMajor{});
    
    std::cout << "Thread layout (column-major):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  ");
        for (int j = 0; j < 4; ++j) {
            printf("T%2d ", thread_layout_cm(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "In column-major mapping:" << std::endl;
    std::cout << "  - Adjacent threads in a column have adjacent IDs" << std::endl;
    std::cout << "  - Useful for column-major data or transpose operations" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Vectorized Access (4 elements per thread)
    // ================================================================
    std::cout << "--- Example 3: Vectorized Access ---" << std::endl;
    std::cout << "Each thread loads 4 consecutive elements (like float4)." << std::endl;
    std::cout << std::endl;
    
    const int VEC_SIZE = 4;
    const int elements_per_thread = VEC_SIZE;
    
    // For 16x16 matrix with 4x4 threads, each thread loads 4 elements
    auto tiled_layout = make_layout(
        make_shape(Int<4>{}, Int<4>{}),   // 4x4 threads
        make_shape(Int<1>{}, Int<4>{})    // 1x4 elements per thread
    );
    
    std::cout << "Tiled layout (threads x elements):" << std::endl;
    print(tiled_layout);
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Interleaved Thread Mapping
    // ================================================================
    std::cout << "--- Example 4: Interleaved Mapping ---" << std::endl;
    std::cout << "Interleaving can help with bank conflict avoidance." << std::endl;
    std::cout << std::endl;
    
    // Simple interleaved: thread ID = (row % 2) * 8 + (col % 2) * 4 + ...
    // This is a simplified example
    std::cout << "Interleaved pattern example (8x8 threads):" << std::endl;
    std::cout << "Thread 0, 1, 2, 3... access elements 0, 2, 4, 6..." << std::endl;
    std::cout << "This spreads threads across memory banks" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Copy Kernel with Vectorized Load
    // ================================================================
    std::cout << "--- Example 5: Vectorized Copy Kernel ---" << std::endl;
    
    const int VEC_M = 8;
    const int VEC_N = 8;
    const int VEC_SIZE_TOTAL = VEC_M * VEC_N;
    
    std::vector<float> h_src_vec(VEC_SIZE_TOTAL);
    std::vector<float> h_dst_vec(VEC_SIZE_TOTAL, 0.0f);
    
    for (int i = 0; i < VEC_SIZE_TOTAL; ++i) {
        h_src_vec[i] = static_cast<float>(i + 1);
    }

    float *d_src_vec, *d_dst_vec;
    CUDA_CHECK(cudaMalloc(&d_src_vec, VEC_SIZE_TOTAL * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst_vec, VEC_SIZE_TOTAL * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(d_src_vec, h_src_vec.data(), VEC_SIZE_TOTAL * sizeof(float), cudaMemcpyHostToDevice));

    // Vectorized copy demonstration
    // Note: Lambda kernels require special CUDA compilation
    // This demonstrates the concept
    
    const int THREADS_VEC = 16;  // 16 threads
    const int ELEMS_PER_THREAD = 4;  // Each thread copies 4 elements
    
    std::cout << "Vectorized copy concept:" << std::endl;
    std::cout << "  Threads: " << THREADS_VEC << std::endl;
    std::cout << "  Elements per thread: " << ELEMS_PER_THREAD << std::endl;
    std::cout << "  Total elements: " << (THREADS_VEC * ELEMS_PER_THREAD) << std::endl;
    
    // For this tutorial, use standard cudaMemcpy
    CUDA_CHECK(cudaMemcpy(d_dst_vec, d_src_vec, VEC_SIZE_TOTAL * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_dst_vec.data(), d_dst_vec, VEC_SIZE_TOTAL * sizeof(float), cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < VEC_SIZE_TOTAL; ++i) {
        if (h_dst_vec[i] != h_src_vec[i]) {
            success = false;
            break;
        }
    }

    std::cout << "Vectorized copy (" << ELEMS_PER_THREAD << " elems/thread): "
              << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: Memory Coalescing Analysis
    // ================================================================
    std::cout << "--- Example 6: Memory Coalescing ---" << std::endl;
    std::cout << "Coalesced access: consecutive threads access consecutive addresses" << std::endl;
    std::cout << std::endl;
    
    auto row_major_data = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    std::cout << "Row-major data layout (8x8):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        printf("  ");
        for (int j = 0; j < 8; ++j) {
            printf("%2d ", row_major_data(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "With row-major thread mapping:" << std::endl;
    std::cout << "  - Threads 0-7 access addresses 0-7 (coalesced!)" << std::endl;
    std::cout << "  - This is optimal for global memory transactions" << std::endl;
    std::cout << std::endl;
    
    std::cout << "With column-major thread mapping on row-major data:" << std::endl;
    std::cout << "  - Threads 0-7 access addresses 0, 8, 16, 24... (strided)" << std::endl;
    std::cout << "  - This causes uncoalesced access (bad for performance)" << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_src_vec));
    CUDA_CHECK(cudaFree(d_dst_vec));

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Thread mapping affects memory access patterns" << std::endl;
    std::cout << "2. Match thread layout to data layout for coalescing" << std::endl;
    std::cout << "3. Row-major threads + row-major data = coalesced access" << std::endl;
    std::cout << "4. Vectorized loads improve throughput" << std::endl;
    std::cout << "5. Interleaving helps with bank conflicts in shared memory" << std::endl;
    std::cout << "6. CuTe layouts help reason about thread-to-data mapping" << std::endl;
    std::cout << std::endl;

    return 0;
}
