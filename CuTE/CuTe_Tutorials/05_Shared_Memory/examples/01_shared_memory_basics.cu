/**
 * @file 01_shared_memory_basics.cu
 * @brief Introduction to Shared Memory in CuTe
 * 
 * This tutorial covers shared memory fundamentals:
 * - What is shared memory and why use it
 * - Shared memory layout creation
 * - Bank conflicts and their impact
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

// Kernel to demonstrate shared memory usage
__global__ void shared_memory_copy_kernel(const float* input, float* output, int size) {
    // Declare shared memory
    extern __shared__ float shared_mem[];
    
    int tid = threadIdx.x;
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load from global to shared memory
    if (global_idx < size) {
        shared_mem[tid] = input[global_idx];
    }
    __syncthreads();
    
    // Load from shared memory and process
    if (global_idx < size) {
        output[global_idx] = shared_mem[tid] * 2.0f;  // Simple operation
    }
}

int main() {
    std::cout << "=== CuTe Tutorial: Shared Memory Basics ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: What is Shared Memory?
    // ================================================================
    std::cout << "--- Concept 1: What is Shared Memory? ---" << std::endl;
    std::cout << "Shared memory is fast on-chip memory shared by all threads in a block." << std::endl;
    std::cout << std::endl;
    std::cout << "Characteristics:" << std::endl;
    std::cout << "  - Much faster than global memory (~100x bandwidth)" << std::endl;
    std::cout << "  - Limited size (typically 48-96 KB per SM)" << std::endl;
    std::cout << "  - Shared among threads in a thread block" << std::endl;
    std::cout << "  - Requires explicit synchronization (__syncthreads)" << std::endl;
    std::cout << std::endl;
    std::cout << "Use cases:" << std::endl;
    std::cout << "  - Caching frequently accessed data" << std::endl;
    std::cout << "  - Thread communication within a block" << std::endl;
    std::cout << "  - Staging data for coalesced global memory access" << std::endl;
    std::cout << "  - Matrix tiles in GEMM" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Shared Memory Layout
    // ================================================================
    std::cout << "--- Example 1: Shared Memory Layout ---" << std::endl;
    
    // Create a layout for shared memory (8x8 tile)
    const int SMEM_M = 8;
    const int SMEM_N = 8;
    
    auto smem_layout = make_layout(make_shape(Int<SMEM_M>{}, Int<SMEM_N>{}), GenRowMajor{});
    
    std::cout << "Shared memory tile: " << SMEM_M << "x" << SMEM_N << std::endl;
    std::cout << "Layout: ";
    print(smem_layout);
    std::cout << std::endl;
    
    std::cout << "Memory offsets:" << std::endl;
    for (int i = 0; i < SMEM_M; ++i) {
        printf("  ");
        for (int j = 0; j < SMEM_N; ++j) {
            printf("%2d ", smem_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 2: Bank Conflicts Introduction
    // ================================================================
    std::cout << "--- Example 2: Bank Conflicts ---" << std::endl;
    std::cout << "Shared memory is divided into banks (typically 32)." << std::endl;
    std::cout << "Each bank can serve one 4-byte access per cycle." << std::endl;
    std::cout << std::endl;
    std::cout << "Bank conflict occurs when multiple threads access the same bank." << std::endl;
    std::cout << "This serializes accesses and reduces bandwidth." << std::endl;
    std::cout << std::endl;
    
    std::cout << "For 32 banks with 4-byte words:" << std::endl;
    std::cout << "  - Bank = (address / 4) % 32" << std::endl;
    std::cout << "  - Consecutive addresses map to consecutive banks" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Visualizing Bank Mapping
    // ================================================================
    std::cout << "--- Example 3: Bank Mapping Visualization ---" << std::endl;
    
    const int BANKS = 32;
    const int ELEMENTS = 64;
    
    std::cout << "For a 64-element shared memory array:" << std::endl;
    std::cout << "Element -> Bank mapping (first 32 elements):" << std::endl;
    
    for (int i = 0; i < 32; ++i) {
        int bank = i % BANKS;
        printf("  Element %2d -> Bank %2d\n", i, bank);
    }
    std::cout << std::endl;
    
    std::cout << "Notice: Element 0 and Element 32 both map to Bank 0!" << std::endl;
    std::cout << "This causes a 2-way bank conflict if accessed simultaneously." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Bank Conflict Example
    // ================================================================
    std::cout << "--- Example 4: Bank Conflict Scenario ---" << std::endl;
    
    // Row-major 8x8 layout
    auto row_major_smem = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    std::cout << "Row-major 8x8 shared memory:" << std::endl;
    std::cout << "If each row is accessed by a different thread:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Column 0 access pattern:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int offset = row_major_smem(i, 0);
        int bank = (offset * 4) / 4 % 32;  // Assuming 4-byte elements
        printf("  Row %d, Col 0 -> Offset %2d -> Bank %2d\n", i, offset, bank);
    }
    std::cout << std::endl;
    
    std::cout << "Column 1 access pattern:" << std::endl;
    for (int i = 0; i < 8; ++i) {
        int offset = row_major_smem(i, 1);
        int bank = (offset * 4) / 4 % 32;
        printf("  Row %d, Col 1 -> Offset %2d -> Bank %2d\n", i, offset, bank);
    }
    std::cout << std::endl;

    // ================================================================
    // Example 5: Simple Shared Memory Kernel
    // ================================================================
    std::cout << "--- Example 5: Shared Memory Kernel ---" << std::endl;
    
    const int SIZE = 256;
    const int BLOCK_SIZE = 128;
    const int NUM_BLOCKS = 2;
    
    std::vector<float> h_input(SIZE);
    std::vector<float> h_output(SIZE, 0.0f);
    
    for (int i = 0; i < SIZE; ++i) {
        h_input[i] = static_cast<float>(i + 1);
    }
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, SIZE * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_output, SIZE * sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), SIZE * sizeof(float), cudaMemcpyHostToDevice));
    
    // Launch kernel with shared memory
    size_t smem_size = BLOCK_SIZE * sizeof(float);
    shared_memory_copy_kernel<<<NUM_BLOCKS, BLOCK_SIZE, smem_size>>>(
        d_input, d_output, SIZE
    );
    
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output.data(), d_output, SIZE * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Verify
    bool success = true;
    for (int i = 0; i < SIZE; ++i) {
        if (h_output[i] != h_input[i] * 2.0f) {
            success = false;
            break;
        }
    }
    
    std::cout << "Shared memory copy kernel: " << (success ? "SUCCESS" : "FAILED") << std::endl;
    std::cout << "Processed " << SIZE << " elements using " << NUM_BLOCKS << " blocks" << std::endl;
    std::cout << "Shared memory per block: " << smem_size << " bytes" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: Padding to Avoid Bank Conflicts
    // ================================================================
    std::cout << "--- Example 6: Padding for Bank Conflict Avoidance ---" << std::endl;
    
    // Without padding
    auto no_pad_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    // With padding (add 1 element per row)
    auto padded_layout = make_layout(
        make_shape(Int<8>{}, Int<8>{}),
        make_stride(Int<9>{}, Int<1>{})  // Row stride = 9 instead of 8
    );
    
    std::cout << "Without padding (8x8, stride=[8,1]):" << std::endl;
    std::cout << "  Column 0 banks: ";
    for (int i = 0; i < 8; ++i) {
        printf("%2d ", no_pad_layout(i, 0) % 32);
    }
    std::cout << std::endl;
    
    std::cout << "With padding (8x8, stride=[9,1]):" << std::endl;
    std::cout << "  Column 0 banks: ";
    for (int i = 0; i < 8; ++i) {
        printf("%2d ", padded_layout(i, 0) % 32);
    }
    std::cout << std::endl;
    std::cout << std::endl;
    
    std::cout << "Padding changes the bank mapping to avoid conflicts!" << std::endl;
    std::cout << std::endl;

    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Shared memory is fast on-chip memory for thread blocks" << std::endl;
    std::cout << "2. Use extern __shared__ for dynamic shared memory" << std::endl;
    std::cout << "3. Shared memory is divided into banks (typically 32)" << std::endl;
    std::cout << "4. Bank conflicts occur when multiple threads access same bank" << std::endl;
    std::cout << "5. Padding can help avoid bank conflicts" << std::endl;
    std::cout << "6. Next: Learn about swizzling for automatic conflict avoidance" << std::endl;
    std::cout << std::endl;

    return 0;
}
