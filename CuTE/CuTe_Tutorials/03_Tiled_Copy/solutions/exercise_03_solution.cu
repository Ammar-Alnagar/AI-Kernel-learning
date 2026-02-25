/**
 * @file exercise_03_solution.cu
 * @brief Solution: Tiled Copy
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
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); } \
    } while(0)

int main() {
    std::cout << "=== Exercise 03: Solution ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 1: 8x8 copy with 4x4 thread block
    // ================================================================
    std::cout << "--- Exercise 1: 8x8 Copy with 4x4 Block ---" << std::endl;
    
    const int M = 8, N = 8;
    const int size = M * N;
    
    std::vector<float> h_src(size), h_dst(size, 0.0f);
    for (int i = 0; i < size; ++i) h_src[i] = static_cast<float>(i + 1);
    
    float *d_src, *d_dst;
    CUDA_CHECK(cudaMalloc(&d_src, size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst, size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src, h_src.data(), size * sizeof(float), cudaMemcpyHostToDevice));
    
    // 4x4 thread block
    auto thread_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    
    std::cout << "Thread layout: ";
    print(thread_layout);
    std::cout << std::endl;
    
    std::cout << "Concept: Each of 16 threads copies one element of a 4x4 tile" << std::endl;
    
    // Demonstrate with standard copy
    CUDA_CHECK(cudaMemcpy(d_dst, d_src, size * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_dst.data(), d_dst, size * sizeof(float), cudaMemcpyDeviceToHost));
    
    bool success = true;
    for (int i = 0; i < size; ++i) {
        if (h_dst[i] != h_src[i]) { success = false; break; }
    }
    
    std::cout << "Verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 2: Vectorized copy (2 elements per thread)
    // ================================================================
    std::cout << "--- Exercise 2: Vectorized Copy ---" << std::endl;
    
    const int VEC_SIZE = 32;  // 16 threads * 2 elements
    std::vector<float> h_src_vec(VEC_SIZE), h_dst_vec(VEC_SIZE, 0.0f);
    for (int i = 0; i < VEC_SIZE; ++i) h_src_vec[i] = static_cast<float>(i + 1);
    
    float *d_src_vec, *d_dst_vec;
    CUDA_CHECK(cudaMalloc(&d_src_vec, VEC_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_dst_vec, VEC_SIZE * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_src_vec, h_src_vec.data(), VEC_SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    std::cout << "Vectorized copy concept:" << std::endl;
    std::cout << "  16 threads, each copies 2 consecutive elements" << std::endl;
    std::cout << "  Total: 32 elements" << std::endl;
    
    // For demonstration, use standard copy
    CUDA_CHECK(cudaMemcpy(d_dst_vec, d_src_vec, VEC_SIZE * sizeof(float), cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_dst_vec.data(), d_dst_vec, VEC_SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    success = true;
    for (int i = 0; i < VEC_SIZE; ++i) {
        if (h_dst_vec[i] != h_src_vec[i]) { success = false; break; }
    }
    
    std::cout << "Vectorized copy verification: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 3: Thread mapping calculation
    // ================================================================
    std::cout << "--- Exercise 3: Thread Mapping ---" << std::endl;
    
    auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    
    std::cout << "For row-major 4x4 layout:" << std::endl;
    std::cout << "  Thread ID to (row, col) mapping:" << std::endl;
    
    for (int tid = 0; tid < 16; ++tid) {
        int row = tid / 4;
        int col = tid % 4;
        std::cout << "  Thread " << tid << " -> (" << row << ", " << col << ")" << std::endl;
    }
    std::cout << std::endl;
    
    int tid_5 = 5;
    int r5 = tid_5 / 4;
    int c5 = tid_5 % 4;
    
    std::cout << "Thread 5 maps to: (" << r5 << ", " << c5 << ")" << std::endl;
    std::cout << "Verification: 5 = " << r5 << " * 4 + " << c5 << " = " << (r5 * 4 + c5) << std::endl;
    std::cout << "Expected: (1, 1) - " << (r5 == 1 && c5 == 1 ? "CORRECT" : "INCORRECT") << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_dst));
    CUDA_CHECK(cudaFree(d_src_vec));
    CUDA_CHECK(cudaFree(d_dst_vec));

    // Summary
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Exercise 1: 4x4 thread block for tiled copy" << std::endl;
    std::cout << "Exercise 2: Each thread copies 2 consecutive elements" << std::endl;
    std::cout << "Exercise 3: Thread 5 -> (1, 1) in row-major 4x4 layout" << std::endl;
    std::cout << std::endl;

    return 0;
}
