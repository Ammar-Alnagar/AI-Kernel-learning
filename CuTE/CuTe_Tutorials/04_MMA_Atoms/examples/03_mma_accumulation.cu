/**
 * @file 03_mma_accumulation.cu
 * @brief MMA Accumulation and Multiple Iterations
 * 
 * This tutorial covers:
 * - Accumulator registers in MMA
 * - Multiple K-tiles accumulation
 * - Building up results over iterations
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
    std::cout << "=== CuTe Tutorial: MMA Accumulation ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: MMA Accumulator
    // ================================================================
    std::cout << "--- Concept 1: MMA Accumulator ---" << std::endl;
    std::cout << "MMA operation: D = A * B + C" << std::endl;
    std::cout << "  C is the accumulator (input)" << std::endl;
    std::cout << "  D is the result (output)" << std::endl;
    std::cout << std::endl;
    std::cout << "For GEMM, we typically start with C = 0:" << std::endl;
    std::cout << "  D = A * B + 0 = A * B" << std::endl;
    std::cout << std::endl;
    std::cout << "For multiple K-tiles, we accumulate:" << std::endl;
    std::cout << "  D += A_k * B_k for each k in K_tiles" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Single MMA Operation
    // ================================================================
    std::cout << "--- Example 1: Single MMA Operation ---" << std::endl;
    
    const int M = 4, N = 4, K = 4;
    
    // Create matrices
    std::vector<float> h_A(M * K);
    std::vector<float> h_B(K * N);
    std::vector<float> h_C(M * N, 0.0f);  // Accumulator (initially zero)
    std::vector<float> h_D(M * N, 0.0f);  // Result
    
    // Initialize A and B with simple patterns
    for (int i = 0; i < M * K; ++i) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < K * N; ++i) h_B[i] = static_cast<float>((i + 1) * 0.1f);
    
    std::cout << "Matrix A (" << M << "x" << K << "):" << std::endl;
    for (int i = 0; i < M; ++i) {
        printf("  ");
        for (int j = 0; j < K; ++j) {
            printf("%4.1f ", h_A[i * K + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Matrix B (" << K << "x" << N << "):" << std::endl;
    for (int i = 0; i < K; ++i) {
        printf("  ");
        for (int j = 0; j < N; ++j) {
            printf("%4.1f ", h_B[i * N + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Single MMA: D = A * B + C (C = 0)
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = h_C[i * N + j];  // Start with accumulator
            for (int l = 0; l < K; ++l) {
                sum += h_A[i * K + l] * h_B[l * N + j];
            }
            h_D[i * N + j] = sum;
        }
    }
    
    std::cout << "Accumulator C (zeros):" << std::endl;
    for (int i = 0; i < M; ++i) {
        printf("  ");
        for (int j = 0; j < N; ++j) {
            printf("%4.1f ", h_C[i * N + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Result D = A * B + C:" << std::endl;
    for (int i = 0; i < M; ++i) {
        printf("  ");
        for (int j = 0; j < N; ++j) {
            printf("%6.2f ", h_D[i * N + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 2: Multiple K-Tile Accumulation
    // ================================================================
    std::cout << "--- Example 2: Multiple K-Tile Accumulation ---" << std::endl;
    std::cout << "When K is large, we split it into tiles and accumulate." << std::endl;
    std::cout << std::endl;
    
    const int LARGE_K = 16;
    const int K_TILES = 4;  // 4 tiles of K=4 each
    const int K_PER_TILE = LARGE_K / K_TILES;
    
    std::vector<float> h_A_large(M * LARGE_K);
    std::vector<float> h_B_large(LARGE_K * N);
    std::vector<float> h_D_accum(M * N, 0.0f);
    
    // Initialize large matrices
    for (int i = 0; i < M * LARGE_K; ++i) h_A_large[i] = static_cast<float>((i + 1) * 0.1f);
    for (int i = 0; i < LARGE_K * N; ++i) h_B_large[i] = static_cast<float>((i + 1) * 0.05f);
    
    std::cout << "Large GEMM: D[" << M << "x" << N << "] = A[" << M << "x" << LARGE_K 
              << "] * B[" << LARGE_K << "x" << N << "]" << std::endl;
    std::cout << "Split K into " << K_TILES << " tiles of size " << K_PER_TILE << std::endl;
    std::cout << std::endl;
    
    // Accumulate over K-tiles
    std::cout << "Accumulation process:" << std::endl;
    
    for (int kt = 0; kt < K_TILES; ++kt) {
        int k_start = kt * K_PER_TILE;
        
        // Compute partial product for this K-tile
        std::vector<float> h_D_partial(M * N, 0.0f);
        for (int i = 0; i < M; ++i) {
            for (int j = 0; j < N; ++j) {
                float sum = 0.0f;
                for (int l = 0; l < K_PER_TILE; ++l) {
                    sum += h_A_large[i * LARGE_K + (k_start + l)] * 
                           h_B_large[(k_start + l) * N + j];
                }
                h_D_partial[i * N + j] = sum;
            }
        }
        
        // Accumulate
        for (int i = 0; i < M * N; ++i) {
            h_D_accum[i] += h_D_partial[i];
        }
        
        std::cout << "  K-tile " << kt << " (k=" << k_start << "-" << (k_start + K_PER_TILE - 1) << "):" << std::endl;
        std::cout << "    Partial sum (first element): " << h_D_partial[0] << std::endl;
        std::cout << "    Running total (first element): " << h_D_accum[0] << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Final accumulated result (first row):" << std::endl;
    for (int j = 0; j < N; ++j) {
        printf("  D[0,%d] = %.2f\n", j, h_D_accum[j]);
    }
    std::cout << std::endl;

    // ================================================================
    // Example 3: Verify Against Full Multiply
    // ================================================================
    std::cout << "--- Example 3: Verification ---" << std::endl;
    
    // Compute full multiply for verification
    std::vector<float> h_D_full(M * N, 0.0f);
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < LARGE_K; ++l) {
                sum += h_A_large[i * LARGE_K + l] * h_B_large[l * N + j];
            }
            h_D_full[i * N + j] = sum;
        }
    }
    
    std::cout << "Full GEMM result (first row):" << std::endl;
    for (int j = 0; j < N; ++j) {
        printf("  D[0,%d] = %.2f\n", j, h_D_full[j]);
    }
    std::cout << std::endl;
    
    // Compare
    bool match = true;
    for (int i = 0; i < M * N; ++i) {
        if (std::abs(h_D_accum[i] - h_D_full[i]) > 1e-5f) {
            match = false;
            break;
        }
    }
    
    std::cout << "Tiled accumulation matches full GEMM: " << (match ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Accumulator Initialization Patterns
    // ================================================================
    std::cout << "--- Example 4: Accumulator Initialization ---" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Pattern 1: Zero initialization (most common)" << std::endl;
    std::cout << "  For standard GEMM: D = A * B" << std::endl;
    std::cout << "  Initialize C = 0, then D = A * B + 0" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Pattern 2: Bias addition" << std::endl;
    std::cout << "  For GEMM + bias: D = A * B + bias" << std::endl;
    std::cout << "  Initialize C with bias values" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Pattern 3: Residual connection" << std::endl;
    std::cout << "  For residual: D = A * B + residual" << std::endl;
    std::cout << "  Initialize C with residual tensor" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Pattern 4: Epilogue fusion" << std::endl;
    std::cout << "  For fused operations: D = activation(A * B + C)" << std::endl;
    std::cout << "  Accumulate first, then apply activation" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Register Usage for Accumulator
    // ================================================================
    std::cout << "--- Example 5: Register Usage ---" << std::endl;
    std::cout << std::endl;
    
    const int ACC_M = 16, ACC_N = 16;
    const int ACC_ELEMENTS = ACC_M * ACC_N;
    
    std::cout << "For a " << ACC_M << "x" << ACC_N << " MMA output:" << std::endl;
    std::cout << "  Accumulator elements: " << ACC_ELEMENTS << std::endl;
    std::cout << "  If using FP32: " << (ACC_ELEMENTS * 4) << " bytes" << std::endl;
    std::cout << "  If using FP16: " << (ACC_ELEMENTS * 2) << " bytes" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Register pressure considerations:" << std::endl;
    std::cout << "  - Larger accumulators need more registers" << std::endl;
    std::cout << "  - Affects maximum occupancy" << std::endl;
    std::cout << "  - Trade-off between tile size and parallelism" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. MMA accumulates: D = A * B + C" << std::endl;
    std::cout << "2. For GEMM, typically start with C = 0" << std::endl;
    std::cout << "3. Large K is split into tiles, accumulate results" << std::endl;
    std::cout << "4. Each K-tile: D += A_k * B_k" << std::endl;
    std::cout << "5. Accumulator can hold bias or residual" << std::endl;
    std::cout << "6. Register usage depends on accumulator size" << std::endl;
    std::cout << std::endl;

    return 0;
}
