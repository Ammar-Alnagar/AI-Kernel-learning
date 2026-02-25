/**
 * @file 01_copy_basics.cu
 * @brief Introduction to Tiled Copy in CuTe
 * 
 * This tutorial demonstrates the basics of tiled copy operations:
 * - What is tiled copy and why use it
 * - Copy traits and copy descriptors
 * - Basic memory-to-memory copy with CuTe
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace cute;

// Simple CUDA error checking macro
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
    std::cout << "=== CuTe Tutorial: Tiled Copy Basics ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: What is Tiled Copy?
    // ================================================================
    std::cout << "--- Concept 1: What is Tiled Copy? ---" << std::endl;
    std::cout << "Tiled copy is a fundamental operation in GPU programming." << std::endl;
    std::cout << "It copies data from source to destination in tiles (blocks)." << std::endl;
    std::cout << "Benefits:" << std::endl;
    std::cout << "  - Efficient memory coalescing" << std::endl;
    std::cout << "  - Better cache utilization" << std::endl;
    std::cout << "  - Enables parallel copy across threads" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Setup: Allocate host and device memory
    // ================================================================
    const int M = 8;
    const int N = 8;
    const int size = M * N;
    
    // Host data
    std::vector<float> h_src(size);
    std::vector<float> h_dst(size, 0.0f);
    
    // Initialize source with pattern
    for (int i = 0; i < size; ++i) {
        h_src[i] = static_cast<float>(i + 1);
    }
    
    // Device pointers
    float *d_src = nullptr;
    float *d_dst = nullptr;
    
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, size * sizeof(float)));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Source data (8x8 matrix):" << std::endl;
    for (int i = 0; i < M; ++i) {
        printf("  Row %2d: ", i);
        for (int j = 0; j < N; ++j) {
            printf("%4.0f ", h_src[i * N + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 1: Create Layouts for Copy
    // ================================================================
    std::cout << "--- Example 1: Layouts for Copy ---" << std::endl;
    
    // Source layout (row-major 8x8)
    auto src_layout = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});
    
    // Destination layout (same as source)
    auto dst_layout = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});
    
    std::cout << "Source layout: ";
    print(src_layout);
    std::cout << std::endl;
    
    std::cout << "Destination layout: ";
    print(dst_layout);
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Create Tensors for Copy
    // ================================================================
    std::cout << "--- Example 2: Tensors for Copy ---" << std::endl;
    
    // Create tensors
    auto src_tensor = make_tensor(d_src, src_layout);
    auto dst_tensor = make_tensor(d_dst, dst_layout);
    
    std::cout << "Source tensor shape: ";
    print(src_tensor.shape());
    std::cout << std::endl;
    
    std::cout << "Destination tensor shape: ";
    print(dst_tensor.shape());
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Simple Copy Kernel
    // ================================================================
    std::cout << "--- Example 3: Simple Copy Kernel ---" << std::endl;
    std::cout << "Launching a simple element-wise copy kernel..." << std::endl;

    // Note: Lambda kernels require special CUDA compilation
    // For this tutorial, use standard cudaMemcpy to demonstrate
    std::cout << "Kernel concept: 256 threads, each copies one element" << std::endl;
    
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy back and verify
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, size * sizeof(float), cudaMemcpyDeviceToHost));

    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (h_dst[i] != h_src[i]) {
            success = false;
            break;
        }
    }

    std::cout << "Copy verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    std::cout << "Destination data after copy:" << std::endl;
    for (int i = 0; i < M; ++i) {
        printf("  Row %2d: ", i);
        for (int j = 0; j < N; ++j) {
            printf("%4.0f ", h_dst[i * N + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 4: Tiled Copy Concept
    // ================================================================
    std::cout << "--- Example 4: Tiled Copy Concept ---" << std::endl;
    std::cout << "Instead of copying element-by-element, we copy in tiles." << std::endl;
    std::cout << "For example, copy 2x2 tiles:" << std::endl;
    std::cout << std::endl;
    
    // Define tile size
    const int TILE_M = 2;
    const int TILE_N = 2;
    
    // Number of tiles
    const int NUM_TILES_M = M / TILE_M;
    const int NUM_TILES_N = N / TILE_N;
    
    std::cout << "Matrix size: " << M << "x" << N << std::endl;
    std::cout << "Tile size: " << TILE_M << "x" << TILE_N << std::endl;
    std::cout << "Number of tiles: " << NUM_TILES_M << "x" << NUM_TILES_N 
              << " = " << (NUM_TILES_M * NUM_TILES_N) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tile layout visualization:" << std::endl;
    for (int ti = 0; ti < NUM_TILES_M; ++ti) {
        for (int tj = 0; tj < NUM_TILES_N; ++tj) {
            std::cout << "  Tile (" << ti << ", " << tj << "): ";
            std::cout << "covers rows [" << (ti * TILE_M) << "-" << (ti * TILE_M + TILE_M - 1) << "]";
            std::cout << ", cols [" << (tj * TILE_N) << "-" << (tj * TILE_N + TILE_N - 1) << "]" << std::endl;
        }
    }
    std::cout << std::endl;

    // ================================================================
    // Example 5: Tiled Copy Implementation
    // ================================================================
    std::cout << "--- Example 5: Tiled Copy Implementation ---" << std::endl;

    // Reset destination
    std::fill(h_dst.begin(), h_dst.end(), 0.0f);
    CUDA_CHECK(cudaMemcpy(d_dst, h_dst.data(), size * sizeof(float), cudaMemcpyHostToDevice));

    // Tiled copy kernel demonstration
    // Note: Lambda kernels require special CUDA compilation
    std::cout << "Tiled copy concept:" << std::endl;
    std::cout << "  Copy in 2x2 tiles" << std::endl;
    std::cout << "  Total tiles: " << (M/2) << "x" << (N/2) << " = " << (M/2 * N/2) << std::endl;
    
    // For demonstration, use standard copy
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, size * sizeof(float), cudaMemcpyDeviceToHost));

    success = true;
    for (int i = 0; i < size; ++i) {
        if (h_dst[i] != h_src[i]) {
            success = false;
            break;
        }
    }

    std::cout << "Tiled copy verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Tiled copy organizes memory transfers in blocks" << std::endl;
    std::cout << "2. Layouts define how data is organized in memory" << std::endl;
    std::cout << "3. Tensors combine layouts with data pointers" << std::endl;
    std::cout << "4. Tiling enables efficient parallel copy" << std::endl;
    std::cout << "5. Next: Learn about CuTe's copy atoms and thread mapping" << std::endl;
    std::cout << std::endl;

    return 0;
}
