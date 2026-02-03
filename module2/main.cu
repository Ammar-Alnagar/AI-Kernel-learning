#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cutlass/cutlass.h>
#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/stride.hpp>
#include <cute/atom/copy_atom.hpp>
#include <cute/atom/mma_atom.hpp>

using namespace cute;

/*
 * Module 2: Tiled Copy (Vectorized Global-to-Shared Memory Movement)
 * Composable Memory Access Patterns
 *
 * This kernel demonstrates how to use CuTe's tiled copy mechanisms to efficiently
 * move data between global and shared memory using vectorized operations.
 * We'll show how to define memory access patterns using layout algebra,
 * enabling automatic vectorization and optimal memory bandwidth utilization.
 */

// Device function to demonstrate tiled copy operations
__global__ void tiled_copy_kernel(float* global_input, float* global_output, int M, int N) {
    // Define shared memory for input and output tiles
    __shared__ __align__(16) float smem_input[128];  // Shared memory for input tile, aligned for vectorization
    __shared__ __align__(16) float smem_output[128]; // Shared memory for output tile, aligned for vectorization

    // Thread block and thread indices
    int tid = threadIdx.x;  // Thread ID within block (0-127 for blockDim.x=128)
    int bid = blockIdx.x;   // Block ID

    // Define the tile size for our computation
    // We'll work with 32x32 tiles processed by 128 threads
    // Each thread will handle multiple elements
    constexpr int TILE_M = 32;
    constexpr int TILE_N = 32;

    // Calculate which tile this block is responsible for
    int tile_row_start = bid * TILE_M;
    int tile_col_start = 0; // For simplicity, we'll process the first TILE_N columns

    // Define the layout for the input tile in global memory
    // Shape: 32 rows x 32 columns
    // Stride: N (leading dimension) for rows, 1 for columns (row-major)
    // Use compile-time constants for layout definition
    auto gInputLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                    make_stride(Int<N>{}, Int<1>{}));

    // Define the layout for the same tile in shared memory
    // For coalesced access, we use a simple row-major layout in shared memory
    auto sInputLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                    make_stride(Int<TILE_N>{}, Int<1>{}));

    // Create tensors representing the global and shared memory views
    auto gInput = make_tensor(make_gmem_ptr(global_input + tile_row_start * N + tile_col_start), gInputLayout);
    auto sInput = make_tensor(make_smem_ptr(smem_input), sInputLayout);

    // Define the copy operation using CuTe's copy atom
    // We'll use a simple copy atom that handles vectorized access
    auto copy_atom = Copy_Atom<DefaultCopy, float>{};

    // Create a tiled copy operation that specifies how threads participate in the copy
    // We'll use 128 threads to copy a 32x32 tile (1024 elements total)
    // Each thread copies 8 elements (since 1024/128 = 8)
    auto tiled_copy = make_tiled_copy(
        copy_atom,
        Layout<Shape<_128>,Stride<_1>>{},  // 128 threads in 1D
        Layout<Shape<_8>,_1>{}             // Each thread copies 8 elements
    );

    // Perform the tiled copy from global to shared memory
    // This operation automatically handles vectorization when possible
    copy(tiled_copy, gInput, sInput);

    // Synchronize threads to ensure all data is loaded
    __syncthreads();

    // Now demonstrate the reverse: copy from shared back to global
    // But first, let's do a simple transformation in shared memory
    // (for demonstration purposes, we'll just double the values)
    #pragma unroll
    for (int i = 0; i < size<0>(sInputLayout); ++i) {
        #pragma unroll
        for (int j = 0; j < size<1>(sInputLayout); ++j) {
            sInput(i, j) *= 2.0f;  // Simple transformation
        }
    }

    // Define the output tile layout in global memory
    auto gOutputLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                                     make_stride(Int<N>{}, Int<1>{}));
    auto gOutput = make_tensor(make_gmem_ptr(global_output + tile_row_start * N + tile_col_start), gOutputLayout);

    // Copy transformed data from shared to global output
    copy(tiled_copy, sInput, gOutput);

    // Synchronize before kernel completion
    __syncthreads();
}

// Alternative implementation showing more advanced tiled copy concepts
__global__ void advanced_tiled_copy_kernel(float* global_input, float* global_output, int M, int N) {
    // Define shared memory
    __shared__ __align__(16) float smem_tile[128];

    // Thread and block indices
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    // Define tile dimensions - using a more complex tiling pattern
    // We'll use a 16x16 tile handled by 128 threads (each thread handles 2 elements)
    constexpr int TILE_M = 16;
    constexpr int TILE_N = 16;

    // Calculate which tile this block processes
    int block_row = bid * TILE_M;
    int block_col = 0; // For simplicity, process first TILE_N columns

    // Define the global memory layout for the tile
    auto gLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                               make_stride(Int<N>{}, Int<1>{}));

    // Create the global tensor
    auto gTensor = make_tensor(make_gmem_ptr(global_input + block_row * N + block_col), gLayout);

    // Define the shared memory layout
    auto sLayout = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_N>{}),
                               make_stride(Int<TILE_N>{}, Int<1>{}));

    // Create shared memory tensor
    auto sTensor = make_tensor(make_smem_ptr(smem_tile), sLayout);

    // Define the copy operation with vectorization considerations
    // Use a copy atom that supports vectorized access
    auto copy_atom = Copy_Atom<DefaultCopy, float>{};

    // Create a tiled copy that maps 128 threads to copy 16x16=256 elements
    // Each thread will copy 2 elements (256/128 = 2)
    auto tiled_copy = make_tiled_copy(
        copy_atom,
        Layout<Shape<_128>,Stride<_1>>{},  // 128 threads in 1D
        Layout<Shape<_2>,_1>{}             // Each thread copies 2 elements
    );

    // Perform the copy operation - CuTe handles the vectorization automatically
    // based on the layout compatibility
    copy(tiled_copy, gTensor, sTensor);

    // Synchronize threads
    __syncthreads();

    // Transform data in shared memory (simple multiplication by 2)
    #pragma unroll
    for (int i = 0; i < size<0>(sLayout); ++i) {
        #pragma unroll
        for (int j = 0; j < size<1>(sLayout); ++j) {
            sTensor(i, j) *= 2.0f;
        }
    }

    // Copy back to global memory
    auto gOutput = make_tensor(make_gmem_ptr(global_output + block_row * N + block_col), gLayout);

    copy(tiled_copy, sTensor, gOutput);

    __syncthreads();
}

int main() {
    std::cout << "=== CUTLASS 3.x CuTe Tiled Copy Demo ===" << std::endl;
    std::cout << "Demonstrating vectorized global-to-shared memory movement" << std::endl;

    // Define problem size
    constexpr int M = 64;
    constexpr int N = 64;
    constexpr int SIZE = M * N;

    // Allocate host memory
    std::vector<float> h_input(SIZE);
    std::vector<float> h_output(SIZE, 0.0f);

    // Initialize input data
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = static_cast<float>(i % 100) / 10.0f;  // Simple pattern
    }

    // Allocate device memory
    float *d_input, *d_output;
    cudaMalloc(&d_input, SIZE * sizeof(float));
    cudaMalloc(&d_output, SIZE * sizeof(float));

    // Copy input data to device
    cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel with 1D block of 128 threads
    dim3 block_dim(128);
    dim3 grid_dim((M + 31) / 32);  // One block per 32-row tile

    std::cout << "Launching basic tiled copy kernel..." << std::endl;
    tiled_copy_kernel<<<grid_dim, block_dim>>>(d_input, d_output, M, N);
    cudaDeviceSynchronize();

    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "Kernel launch failed: " << cudaGetErrorString(err) << std::endl;
        return -1;
    }

    // Copy result back to host
    cudaMemcpy(h_output.data(), d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results (first few elements)
    std::cout << "Verification (first 10 elements):" << std::endl;
    for (int i = 0; i < 10; ++i) {
        std::cout << "Input[" << i << "] = " << h_input[i]
                  << ", Output[" << i << "] = " << h_output[i]
                  << ", Expected = " << (h_input[i] * 2.0f) << std::endl;
    }

    // Clean up
    cudaFree(d_input);
    cudaFree(d_output);

    std::cout << "\n=== Key Concepts Demonstrated ===" << std::endl;
    std::cout << "1. Tiled memory access patterns using CuTe layouts" << std::endl;
    std::cout << "2. Vectorized memory operations through layout algebra" << std::endl;
    std::cout << "3. Thread-to-data mapping using mathematical compositions" << std::endl;
    std::cout << "4. Shared memory tiling for efficient data reuse" << std::endl;
    std::cout << "5. Elimination of manual indexing through composable abstractions" << std::endl;

    return 0;
}