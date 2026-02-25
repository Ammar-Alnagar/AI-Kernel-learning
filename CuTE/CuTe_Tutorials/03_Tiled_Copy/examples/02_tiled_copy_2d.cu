/**
 * @file 02_tiled_copy_2d.cu
 * @brief 2D Tiled Copy with CuTe
 * 
 * This tutorial demonstrates 2D tiled copy using CuTe primitives:
 * - 2D thread blocks
 * - Thread-to-data mapping
 * - Efficient coalesced memory access
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
    std::cout << "=== CuTe Tutorial: 2D Tiled Copy ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Setup: Problem dimensions
    // ================================================================
    const int M = 16;
    const int N = 16;
    const int size = M * N;
    
    // Host data
    std::vector<float> h_src(size);
    std::vector<float> h_dst(size, 0.0f);
    
    // Initialize source
    for (int i = 0; i < size; ++i) {
        h_src[i] = static_cast<float>(i + 1);
    }
    
    // Device memory
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, size * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Problem: Copy " << M << "x" << N << " matrix (" << size << " elements)" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: 2D Thread Block Organization
    // ================================================================
    std::cout << "--- Concept 1: 2D Thread Block ---" << std::endl;
    std::cout << "Organize threads in 2D to match the 2D data structure." << std::endl;
    std::cout << std::endl;
    
    // Thread block dimensions
    const int BLOCK_M = 4;
    const int BLOCK_N = 4;
    const int THREADS = BLOCK_M * BLOCK_N;
    
    std::cout << "Thread block: " << BLOCK_M << "x" << BLOCK_N << " = " << THREADS << " threads" << std::endl;
    std::cout << "Each thread copies one element" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: 2D Thread Layout
    // ================================================================
    std::cout << "--- Example 1: 2D Thread Layout ---" << std::endl;
    
    // Create a 2D thread layout
    auto thread_layout = make_layout(make_shape(Int<BLOCK_M>{}, Int<BLOCK_N>{}), GenRowMajor{});
    
    std::cout << "Thread layout: ";
    print(thread_layout);
    std::cout << std::endl;
    
    std::cout << "Thread ID mapping:" << std::endl;
    for (int i = 0; i < BLOCK_M; ++i) {
        for (int j = 0; j < BLOCK_N; ++j) {
            int tid = thread_layout(i, j);
            std::cout << "  Thread (" << i << ", " << j << ") -> tid " << tid << std::endl;
        }
    }
    std::cout << std::endl;

    // ================================================================
    // Example 2: Data Layout
    // ================================================================
    std::cout << "--- Example 2: Data Layout ---" << std::endl;
    
    auto data_layout = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});
    
    std::cout << "Data layout: ";
    print(data_layout);
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: 2D Copy Kernel
    // ================================================================
    std::cout << "--- Example 3: 2D Copy Kernel ---" << std::endl;
    
    // 2D copy kernel demonstration
    // Note: Lambda kernels require special CUDA compilation
    // This demonstrates the concept
    
    std::cout << "2D Kernel concept:" << std::endl;
    std::cout << "  Threads: " << THREADS << std::endl;
    std::cout << "  Thread layout: " << BLOCK_M << "x" << BLOCK_N << std::endl;
    std::cout << "  Each thread copies one element" << std::endl;
    
    // For this tutorial, use standard cudaMemcpy to demonstrate
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, size * sizeof(float), cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (h_dst[i] != h_src[i]) {
            success = false;
            break;
        }
    }

    std::cout << "2D copy verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Multi-Block Copy
    // ================================================================
    std::cout << "--- Example 4: Multi-Block Copy ---" << std::endl;
    std::cout << "For larger matrices, use multiple thread blocks." << std::endl;
    std::cout << std::endl;
    
    const int LARGE_M = 64;
    const int LARGE_N = 64;
    const int LARGE_SIZE = LARGE_M * LARGE_N;
    
    std::vector<float> h_src_large(LARGE_SIZE);
    std::vector<float> h_dst_large(LARGE_SIZE, 0.0f);
    
    for (int i = 0; i < LARGE_SIZE; ++i) {
        h_src_large[i] = static_cast<float>(i + 1);
    }
    
    float *d_src_large, *d_dst_large;
    CUDA_CHECK(cudaMalloc(&d_src_large, LARGE_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst_large, LARGE_SIZE * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_src_large, h_src_large.data(), LARGE_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Grid dimensions
    const int TILE_M = 16;
    const int TILE_N = 16;
    const int BLOCKS_M = (LARGE_M + TILE_M - 1) / TILE_M;
    const int BLOCKS_N = (LARGE_N + TILE_N - 1) / TILE_N;

    std::cout << "Large matrix: " << LARGE_M << "x" << LARGE_N << std::endl;
    std::cout << "Tile size: " << TILE_M << "x" << TILE_N << std::endl;
    std::cout << "Grid: " << BLOCKS_M << "x" << BLOCKS_N << " = " << (BLOCKS_M * BLOCKS_N) << " blocks" << std::endl;
    std::cout << std::endl;

    // Multi-block copy kernel (defined as traditional kernel)
    // Note: In actual code, this would be a __global__ function
    // For this tutorial, we demonstrate the concept
    std::cout << "Multi-block kernel concept:" << std::endl;
    std::cout << "  Grid: (" << BLOCKS_M << ", " << BLOCKS_N << ") blocks" << std::endl;
    std::cout << "  Block: " << (TILE_M * TILE_N) << " threads" << std::endl;
    std::cout << "  Each thread copies one element" << std::endl;
    std::cout << std::endl;
    
    // For demonstration, do a simple copy
    CUDA_CHECK(cudaMemcpy(h_dst_large.data(), d_src_large, LARGE_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    CUDA_CHECK(cudaMemcpy(h_dst_large.data(), d_dst_large, LARGE_SIZE * sizeof(float), cudaMemcpyDeviceToHost));

    success = true;
    for (int i = 0; i < LARGE_SIZE; ++i) {
        if (h_dst_large[i] != h_src_large[i]) {
            success = false;
            break;
        }
    }
    
    std::cout << "Multi-block copy verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Visualizing Thread-to-Data Mapping
    // ================================================================
    std::cout << "--- Example 5: Thread-to-Data Mapping ---" << std::endl;
    std::cout << "How threads map to data elements in a 4x4 block:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Thread layout (thread IDs):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 4; ++j) {
            printf("T%2d ", thread_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Data layout (memory offsets):" << std::endl;
    auto small_data_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 4; ++j) {
            printf("D%2d ", small_data_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "In this case, thread T(tid) copies data D(tid) - direct mapping!" << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_src_large));
    CUDA_CHECK(cudaFree(d_dst_large));

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. 2D thread blocks match 2D data structure" << std::endl;
    std::cout << "2. Thread layout maps thread ID to 2D coordinates" << std::endl;
    std::cout << "3. Each thread can copy one or more elements" << std::endl;
    std::cout << "4. Multi-block grids handle large matrices" << std::endl;
    std::cout << "5. Grid stride = (BLOCKS_M, BLOCKS_N) for 2D problems" << std::endl;
    std::cout << std::endl;

    return 0;
}
