/**
 * @file 01_gemm_pattern.cu
 * @brief Complete GEMM Pattern with CuTe
 * 
 * This tutorial demonstrates a complete GEMM implementation pattern:
 * - Thread block organization
 * - Shared memory tiling
 * - MMA accumulation
 * - Full kernel structure
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>
#include <cmath>

using namespace cute;

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); } \
    } while(0)

// Simple reference GEMM on CPU
void reference_gemm(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[m * K + k] * B[k * N + n];
            }
            C[m * N + n] = sum;
        }
    }
}

int main() {
    std::cout << "=== CuTe Tutorial: GEMM Pattern ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: GEMM Problem Structure
    // ================================================================
    std::cout << "--- Concept 1: GEMM Structure ---" << std::endl;
    std::cout << "GEMM: C[MxN] = A[MxK] * B[KxN]" << std::endl;
    std::cout << std::endl;
    std::cout << "Key components:" << std::endl;
    std::cout << "  - Thread blocks compute output tiles" << std::endl;
    std::cout << "  - Shared memory stores input tiles" << std::endl;
    std::cout << "  - MMA atoms compute partial products" << std::endl;
    std::cout << "  - Accumulate over K dimension" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: GEMM Configuration
    // ================================================================
    std::cout << "--- Example 1: GEMM Configuration ---" << std::endl;
    
    const int M = 64;
    const int N = 64;
    const int K = 64;
    
    // Tile sizes
    const int BLOCK_M = 16;
    const int BLOCK_N = 16;
    const int BLOCK_K = 16;
    
    // Grid and block dimensions
    const int GRID_M = (M + BLOCK_M - 1) / BLOCK_M;
    const int GRID_N = (N + BLOCK_N - 1) / BLOCK_N;
    const int THREADS = BLOCK_M * BLOCK_N;
    
    std::cout << "Problem: C[" << M << "x" << N << "] = A[" << M << "x" << K 
              << "] * B[" << K << "x" << N << "]" << std::endl;
    std::cout << std::endl;
    std::cout << "Tile size: " << BLOCK_M << "x" << BLOCK_N << "x" << BLOCK_K << std::endl;
    std::cout << "Grid: " << GRID_M << "x" << GRID_N << " = " << (GRID_M * GRID_N) << " blocks" << std::endl;
    std::cout << "Threads per block: " << THREADS << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Layout Setup
    // ================================================================
    std::cout << "--- Example 2: Layout Setup ---" << std::endl;
    
    // Global memory layouts
    auto A_layout = make_layout(make_shape(Int<M>{}, Int<K>{}), GenRowMajor{});
    auto B_layout = make_layout(make_shape(Int<K>{}, Int<N>{}), GenColMajor{});
    auto C_layout = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});
    
    std::cout << "Global memory layouts:" << std::endl;
    std::cout << "  A (row-major): "; print(A_layout); std::cout << std::endl;
    std::cout << "  B (col-major): "; print(B_layout); std::cout << std::endl;
    std::cout << "  C (row-major): "; print(C_layout); std::cout << std::endl;
    std::cout << std::endl;
    
    // Shared memory layouts (with padding)
    auto As_layout = make_layout(
        make_shape(Int<BLOCK_M>{}, Int<BLOCK_K>{}),
        make_stride(Int<BLOCK_K + 1>{}, Int<1>{})
    );
    auto Bs_layout = make_layout(
        make_shape(Int<BLOCK_K>{}, Int<BLOCK_N>{}),
        make_stride(Int<1>{}, Int<BLOCK_K + 1>{})
    );
    
    std::cout << "Shared memory layouts (with padding):" << std::endl;
    std::cout << "  As: "; print(As_layout); std::cout << std::endl;
    std::cout << "  Bs: "; print(Bs_layout); std::cout << std::endl;
    std::cout << std::endl;
    
    int smem_size = (BLOCK_M * (BLOCK_K + 1) + (BLOCK_K + 1) * BLOCK_N) * sizeof(float);
    std::cout << "Shared memory per block: " << smem_size << " bytes" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Reference Implementation
    // ================================================================
    std::cout << "--- Example 3: Reference GEMM ---" << std::endl;
    
    // Allocate host memory
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C_ref(M * N);
    std::vector<float> h_C(M * N, 0.0f);
    
    // Initialize with simple patterns
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>((i % 10) + 1) * 0.1f;
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>((i % 7) + 1) * 0.1f;
    
    // Compute reference
    reference_gemm(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);
    
    std::cout << "Reference GEMM computed." << std::endl;
    std::cout << "Sample output (first row of C):" << std::endl;
    for (int j = 0; j < 8 && j < N; ++j) {
        printf("  C[0,%d] = %.4f\n", j, h_C_ref[j]);
    }
    std::cout << std::endl;

    // ================================================================
    // Example 4: Tiled GEMM Visualization
    // ================================================================
    std::cout << "--- Example 4: Tiled GEMM Visualization ---" << std::endl;
    
    std::cout << "Output tile computation:" << std::endl;
    std::cout << std::endl;
    
    for (int tile_m = 0; tile_m < GRID_M && tile_m < 4; ++tile_m) {
        for (int tile_n = 0; tile_n < GRID_N && tile_n < 4; ++tile_n) {
            int tile_idx = tile_m * GRID_N + tile_n;
            printf("  Tile (%d,%d) -> Block %d\n", tile_m, tile_n, tile_idx);
        }
    }
    std::cout << std::endl;
    
    std::cout << "Each block computes one " << BLOCK_M << "x" << BLOCK_N << " output tile." << std::endl;
    std::cout << "K dimension is split into " << (K / BLOCK_K) << " iterations." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: K-Tile Accumulation
    // ================================================================
    std::cout << "--- Example 5: K-Tile Accumulation ---" << std::endl;
    
    const int K_TILES = K / BLOCK_K;
    
    std::cout << "For each output tile, accumulate " << K_TILES << " K-tiles:" << std::endl;
    std::cout << std::endl;
    std::cout << "  C_tile = 0" << std::endl;
    for (int kt = 0; kt < K_TILES; ++kt) {
        std::cout << "  C_tile += A_tile[" << kt << "] * B_tile[" << kt << "]" << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "This is the core GEMM accumulation pattern." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: Performance Considerations
    // ================================================================
    std::cout << "--- Example 6: Performance Considerations ---" << std::endl;
    
    std::cout << "Key optimizations for GEMM:" << std::endl;
    std::cout << std::endl;
    std::cout << "1. Memory Coalescing:" << std::endl;
    std::cout << "   - Threads access consecutive addresses" << std::endl;
    std::cout << "   - Maximizes global memory bandwidth" << std::endl;
    std::cout << std::endl;
    
    std::cout << "2. Shared Memory Reuse:" << std::endl;
    std::cout << "   - Load tiles once, use multiple times" << std::endl;
    std::cout << "   - Reduces global memory traffic" << std::endl;
    std::cout << std::endl;
    
    std::cout << "3. Bank Conflict Avoidance:" << std::endl;
    std::cout << "   - Use padding or swizzling" << std::endl;
    std::cout << "   - Prevents serialization" << std::endl;
    std::cout << std::endl;
    
    std::cout << "4. MMA Utilization:" << std::endl;
    std::cout << "   - Use tensor cores for acceleration" << std::endl;
    std::cout << "   - Proper operand layouts" << std::endl;
    std::cout << std::endl;
    
    std::cout << "5. Occupancy:" << std::endl;
    std::cout << "   - Balance register usage and parallelism" << std::endl;
    std::cout << "   - Hide memory latency" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. GEMM tiles the output matrix across thread blocks" << std::endl;
    std::cout << "2. Each block loads A and B tiles into shared memory" << std::endl;
    std::cout << "3. Accumulate partial products over K dimension" << std::endl;
    std::cout << "4. Use padding/swizzling to avoid bank conflicts" << std::endl;
    std::cout << "5. MMA atoms accelerate the core multiply" << std::endl;
    std::cout << "6. Next: Software pipelining for latency hiding" << std::endl;
    std::cout << std::endl;

    return 0;
}
