/**
 * @file 03_bank_conflict_free.cu
 * @brief Designing Bank-Conflict-Free Shared Memory Layouts
 * 
 * This tutorial demonstrates:
 * - Analyzing bank conflicts
 * - Designing conflict-free layouts
 * - Practical GEMM shared memory patterns
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

// Kernel demonstrating bank-conflict-free shared memory access
__global__ void gemm_smem_kernel(const float* A, const float* B, float* C, 
                                  int M, int N, int K,
                                  int tile_M, int tile_N, int tile_K) {
    // Shared memory for tiles
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = &smem[tile_M * tile_K];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    
    float acc = 0.0f;
    
    // Loop over K tiles
    for (int t = 0; t < K / tile_K; ++t) {
        // Load A tile (coalesced)
        int a_row = blockIdx.y * tile_M + ty;
        int a_col = t * tile_K + tx;
        if (a_row < M && a_col < K) {
            As[ty * tile_K + tx] = A[a_row * K + a_col];
        }
        
        // Load B tile (coalesced)
        int b_row = t * tile_K + ty;
        int b_col = blockIdx.x * tile_N + tx;
        if (b_row < K && b_col < N) {
            Bs[ty * tile_N + tx] = B[b_row * N + b_col];
        }
        
        __syncthreads();
        
        // Compute partial dot product
        for (int k = 0; k < tile_K; ++k) {
            acc += As[ty * tile_K + k] * Bs[k * tile_N + tx];
        }
        
        __syncthreads();
    }
    
    // Store result
    int c_row = blockIdx.y * tile_M + ty;
    int c_col = blockIdx.x * tile_N + tx;
    if (c_row < M && c_col < N) {
        C[c_row * N + c_col] = acc;
    }
}

int main() {
    std::cout << "=== CuTe Tutorial: Bank-Conflict-Free Layouts ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: Analyzing Bank Conflicts
    // ================================================================
    std::cout << "--- Concept 1: Analyzing Bank Conflicts ---" << std::endl;
    std::cout << "To analyze bank conflicts:" << std::endl;
    std::cout << "  1. Identify which addresses are accessed simultaneously" << std::endl;
    std::cout << "  2. Map addresses to banks: bank = (address / element_size) % num_banks" << std::endl;
    std::cout << "  3. Count how many threads access each bank" << std::endl;
    std::cout << "  4. Max count = conflict degree (1 = no conflict)" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Padding Strategy
    // ================================================================
    std::cout << "--- Example 1: Padding Strategy ---" << std::endl;
    
    const int TILE_M = 16;
    const int TILE_K = 16;
    const int NUM_BANKS = 32;
    
    // Without padding
    auto no_pad = make_layout(make_shape(Int<TILE_M>{}, Int<TILE_K>{}), GenRowMajor{});
    
    // With padding (add 1 element per row)
    auto padded = make_layout(
        make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
        make_stride(Int<TILE_K + 1>{}, Int<1>{})
    );
    
    std::cout << "16x16 tile with 32 banks:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Column access pattern (vertical stride):" << std::endl;
    std::cout << "Without padding:" << std::endl;
    for (int row = 0; row < TILE_M; ++row) {
        int offset = no_pad(row, 0);
        int bank = offset % NUM_BANKS;
        printf("  Row %2d: offset=%3d, bank=%2d\n", row, offset, bank);
    }
    std::cout << std::endl;
    
    std::cout << "With padding (stride=17):" << std::endl;
    for (int row = 0; row < TILE_M; ++row) {
        int offset = padded(row, 0);
        int bank = offset % NUM_BANKS;
        printf("  Row %2d: offset=%3d, bank=%2d\n", row, offset, bank);
    }
    std::cout << std::endl;
    
    std::cout << "Padding spreads accesses across more banks!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Conflict Analysis Function
    // ================================================================
    std::cout << "--- Example 2: Conflict Analysis ---" << std::endl;
    
    auto analyze_conflicts = [](auto layout, int num_banks, const char* name) {
        std::vector<int> bank_counts(num_banks, 0);
        
        // Analyze column access (worst case for row-major)
        for (int col = 0; col < get<1>(layout.shape()); ++col) {
            std::vector<int> banks_accessed;
            for (int row = 0; row < get<0>(layout.shape()); ++row) {
                int offset = layout(row, col);
                int bank = offset % num_banks;
                bank_counts[bank]++;
                banks_accessed.push_back(bank);
            }
        }
        
        int max_conflict = 0;
        for (int b = 0; b < num_banks; ++b) {
            if (bank_counts[b] > max_conflict) {
                max_conflict = bank_counts[b];
            }
        }
        
        std::cout << name << ":" << std::endl;
        std::cout << "  Max conflict degree: " << max_conflict << std::endl;
        std::cout << "  Banks used: ";
        int banks_used = 0;
        for (int b = 0; b < num_banks; ++b) {
            if (bank_counts[b] > 0) banks_used++;
        }
        std::cout << banks_used << " / " << num_banks << std::endl;
    };
    
    const int BANKS = 32;
    
    analyze_conflicts(no_pad, BANKS, "Without padding");
    analyze_conflicts(padded, BANKS, "With padding");
    std::cout << std::endl;

    // ================================================================
    // Example 3: GEMM Shared Memory Layout
    // ================================================================
    std::cout << "--- Example 3: GEMM Shared Memory Layout ---" << std::endl;
    
    const int GEMM_M = 16;
    const int GEMM_N = 16;
    const int GEMM_K = 16;
    
    // A tile: M x K (row-major with padding)
    auto A_smem = make_layout(
        make_shape(Int<GEMM_M>{}, Int<GEMM_K>{}),
        make_stride(Int<GEMM_K + 1>{}, Int<1>{})
    );
    
    // B tile: K x N (column-major with padding)
    auto B_smem = make_layout(
        make_shape(Int<GEMM_K>{}, Int<GEMM_N>{}),
        make_stride(Int<1>{}, Int<GEMM_K + 1>{})
    );
    
    std::cout << "GEMM tile: " << GEMM_M << "x" << GEMM_N << "x" << GEMM_K << std::endl;
    std::cout << std::endl;
    
    std::cout << "A shared memory layout (MxK with padding):" << std::endl;
    std::cout << "  Shape:  "; print(A_smem.shape()); std::cout << std::endl;
    std::cout << "  Stride: "; print(A_smem.stride()); std::cout << std::endl;
    std::cout << "  Size: " << (GEMM_M * (GEMM_K + 1)) << " elements" << std::endl;
    std::cout << std::endl;
    
    std::cout << "B shared memory layout (KxN with padding):" << std::endl;
    std::cout << "  Shape:  "; print(B_smem.shape()); std::cout << std::endl;
    std::cout << "  Stride: "; print(B_smem.stride()); std::cout << std::endl;
    std::cout << "  Size: " << ((GEMM_K + 1) * GEMM_N) << " elements" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Thread-to-Data Mapping
    // ================================================================
    std::cout << "--- Example 4: Thread-to-Data Mapping ---" << std::endl;
    
    const int THREADS_M = 4;
    const int THREADS_N = 4;
    
    auto thread_layout = make_layout(
        make_shape(Int<THREADS_M>{}, Int<THREADS_N>{}),
        GenRowMajor{}
    );
    
    std::cout << "Thread block: " << THREADS_M << "x" << THREADS_N 
              << " = " << (THREADS_M * THREADS_N) << " threads" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Thread layout:" << std::endl;
    for (int i = 0; i < THREADS_M; ++i) {
        printf("  ");
        for (int j = 0; j < THREADS_N; ++j) {
            printf("T%2d ", thread_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Each thread loads/stores one element per iteration." << std::endl;
    std::cout << "With proper layout, all threads access different banks." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Complete GEMM Shared Memory Size
    // ================================================================
    std::cout << "--- Example 5: Shared Memory Size Calculation ---" << std::endl;
    
    int smem_A_size = GEMM_M * (GEMM_K + 1);  // A tile with padding
    int smem_B_size = (GEMM_K + 1) * GEMM_N;  // B tile with padding
    int total_smem_elements = smem_A_size + smem_B_size;
    int total_smem_bytes = total_smem_elements * sizeof(float);
    
    std::cout << "For " << GEMM_M << "x" << GEMM_N << "x" << GEMM_K << " GEMM tile:" << std::endl;
    std::cout << "  A tile: " << GEMM_M << "x" << (GEMM_K + 1) << " = " << smem_A_size << " elements" << std::endl;
    std::cout << "  B tile: " << (GEMM_K + 1) << "x" << GEMM_N << " = " << smem_B_size << " elements" << std::endl;
    std::cout << "  Total: " << total_smem_elements << " elements" << std::endl;
    std::cout << "  Bytes: " << total_smem_bytes << " bytes" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Typical shared memory per SM: 48-96 KB" << std::endl;
    std::cout << "This tile uses: " << (total_smem_bytes / 1024.0) << " KB" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Analyze conflicts by mapping simultaneous accesses to banks" << std::endl;
    std::cout << "2. Padding is a simple way to avoid conflicts" << std::endl;
    std::cout << "3. Add 1 element padding per row for row-major layouts" << std::endl;
    std::cout << "4. GEMM needs padded A and B tiles in shared memory" << std::endl;
    std::cout << "5. Thread-to-data mapping affects conflict patterns" << std::endl;
    std::cout << "6. Swizzling provides automatic conflict avoidance" << std::endl;
    std::cout << std::endl;

    return 0;
}
