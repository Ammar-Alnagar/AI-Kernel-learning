/**
 * @file 01_mma_introduction.cu
 * @brief Introduction to MMA (Matrix Multiply-Accumulate) Atoms
 * 
 * This tutorial introduces MMA operations in CuTe:
 * - What are MMA atoms
 * - MMA traits and descriptors
 * - Basic MMA operation setup
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
    std::cout << "=== CuTe Tutorial: MMA Introduction ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: What is MMA?
    // ================================================================
    std::cout << "--- Concept 1: What is MMA? ---" << std::endl;
    std::cout << "MMA = Matrix Multiply-Accumulate" << std::endl;
    std::cout << "D = A * B + C (matrix multiplication with accumulation)" << std::endl;
    std::cout << std::endl;
    std::cout << "MMA Atoms are hardware-accelerated tensor core operations." << std::endl;
    std::cout << "They perform small matrix multiplications very efficiently." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 2: MMA Shapes
    // ================================================================
    std::cout << "--- Concept 2: MMA Shapes ---" << std::endl;
    std::cout << "Common MMA tensor core shapes (Volta/Ampere):" << std::endl;
    std::cout << "  - 16x8x16  (16 rows A, 8 cols A / 16 rows B, 16 cols B)" << std::endl;
    std::cout << "  - 16x16x16 (square multiply)" << std::endl;
    std::cout << "  - 32x8x16  (wider output)" << std::endl;
    std::cout << std::endl;
    std::cout << "For an MMA operation D = A * B:" << std::endl;
    std::cout << "  A: M x K matrix" << std::endl;
    std::cout << "  B: K x N matrix" << std::endl;
    std::cout << "  D: M x N matrix (result)" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: MMA Problem Setup
    // ================================================================
    std::cout << "--- Example 1: MMA Problem Setup ---" << std::endl;
    
    // MMA dimensions
    const int M = 16;  // Output rows
    const int N = 16;  // Output columns
    const int K = 16;  // Reduction dimension
    
    std::cout << "MMA Problem: D[" << M << "x" << N << "] = A[" << M << "x" << K 
              << "] * B[" << K << "x" << N << "]" << std::endl;
    std::cout << std::endl;
    
    // Create layouts for A, B, and D matrices
    auto layout_A = make_layout(make_shape(Int<M>{}, Int<K>{}), GenRowMajor{});
    auto layout_B = make_layout(make_shape(Int<K>{}, Int<N>{}), GenColMajor{});
    auto layout_D = make_layout(make_shape(Int<M>{}, Int<N>{}), GenRowMajor{});
    
    std::cout << "Layout A (MxK, row-major): ";
    print(layout_A);
    std::cout << std::endl;
    
    std::cout << "Layout B (KxN, col-major): ";
    print(layout_B);
    std::cout << std::endl;
    
    std::cout << "Layout D (MxN, row-major): ";
    print(layout_D);
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: MMA Operand Layouts
    // ================================================================
    std::cout << "--- Example 2: MMA Operand Layouts ---" << std::endl;
    std::cout << "For tensor core MMA, operands have specific layout requirements:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Matrix A (left operand):" << std::endl;
    std::cout << "  - Shape: M x K" << std::endl;
    std::cout << "  - Typically row-major or special tensor core layout" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Matrix B (right operand):" << std::endl;
    std::cout << "  - Shape: K x N" << std::endl;
    std::cout << "  - Typically column-major or special tensor core layout" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Matrix D (accumulator/output):" << std::endl;
    std::cout << "  - Shape: M x N" << std::endl;
    std::cout << "  - Matches the output dimensions" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Simple Matrix Multiply (CPU Reference)
    // ================================================================
    std::cout << "--- Example 3: Matrix Multiply Reference ---" << std::endl;
    
    // Small example for demonstration
    const int m = 4, n = 4, k = 4;
    
    std::vector<float> h_A(m * k);
    std::vector<float> h_B(k * n);
    std::vector<float> h_D(m * n, 0.0f);
    
    // Initialize A and B
    for (int i = 0; i < m * k; ++i) h_A[i] = static_cast<float>(i + 1);
    for (int i = 0; i < k * n; ++i) h_B[i] = static_cast<float>((i + 1) * 0.5f);
    
    std::cout << "Matrix A (" << m << "x" << k << "):" << std::endl;
    for (int i = 0; i < m; ++i) {
        printf("  ");
        for (int j = 0; j < k; ++j) {
            printf("%5.1f ", h_A[i * k + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Matrix B (" << k << "x" << n << "):" << std::endl;
    for (int i = 0; i < k; ++i) {
        printf("  ");
        for (int j = 0; j < n; ++j) {
            printf("%5.1f ", h_B[i * n + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // Reference matrix multiply: D = A * B
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = 0.0f;
            for (int l = 0; l < k; ++l) {
                sum += h_A[i * k + l] * h_B[l * n + j];
            }
            h_D[i * n + j] = sum;
        }
    }
    
    std::cout << "Result D = A * B (" << m << "x" << n << "):" << std::endl;
    for (int i = 0; i < m; ++i) {
        printf("  ");
        for (int j = 0; j < n; ++j) {
            printf("%7.1f ", h_D[i * n + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 4: MMA with Accumulation
    // ================================================================
    std::cout << "--- Example 4: MMA with Accumulation ---" << std::endl;
    std::cout << "MMA operation: D = A * B + C (where C is the accumulator)" << std::endl;
    std::cout << std::endl;
    
    std::vector<float> h_C(m * n);
    for (int i = 0; i < m * n; ++i) {
        h_C[i] = static_cast<float>(i + 1);  // Initial accumulator values
    }
    
    std::cout << "Initial accumulator C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        printf("  ");
        for (int j = 0; j < n; ++j) {
            printf("%5.1f ", h_C[i * n + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    // D = A * B + C
    std::vector<float> h_D_acc(m * n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            float sum = h_C[i * n + j];  // Start with accumulator
            for (int l = 0; l < k; ++l) {
                sum += h_A[i * k + l] * h_B[l * n + j];
            }
            h_D_acc[i * n + j] = sum;
        }
    }
    
    std::cout << "Result D = A * B + C:" << std::endl;
    for (int i = 0; i < m; ++i) {
        printf("  ");
        for (int j = 0; j < n; ++j) {
            printf("%7.1f ", h_D_acc[i * n + j]);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 5: Tiled MMA Concept
    // ================================================================
    std::cout << "--- Example 5: Tiled MMA ---" << std::endl;
    std::cout << "Large matrix multiplies are tiled into smaller MMA operations." << std::endl;
    std::cout << std::endl;
    
    const int LARGE_M = 64, LARGE_N = 64, LARGE_K = 64;
    const int TILE_M = 16, TILE_N = 16, TILE_K = 16;
    
    int tiles_m = (LARGE_M + TILE_M - 1) / TILE_M;
    int tiles_n = (LARGE_N + TILE_N - 1) / TILE_N;
    int tiles_k = (LARGE_K + TILE_K - 1) / TILE_K;
    
    std::cout << "Large GEMM: " << LARGE_M << "x" << LARGE_N << " = " 
              << LARGE_M << "x" << LARGE_K << " * " << LARGE_K << "x" << LARGE_N << std::endl;
    std::cout << "MMA tile: " << TILE_M << "x" << TILE_N << "x" << TILE_K << std::endl;
    std::cout << "Grid: " << tiles_m << "x" << tiles_n << "x" << tiles_k 
              << " = " << (tiles_m * tiles_n * tiles_k) << " MMA operations" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. MMA = Matrix Multiply-Accumulate (D = A*B + C)" << std::endl;
    std::cout << "2. Tensor cores accelerate small matrix multiplies" << std::endl;
    std::cout << "3. Common MMA shapes: 16x8x16, 16x16x16, 32x8x16" << std::endl;
    std::cout << "4. A is MxK, B is KxN, D is MxN" << std::endl;
    std::cout << "5. Large GEMMs are tiled into multiple MMA operations" << std::endl;
    std::cout << "6. Next: Learn about MMA atom shapes and execution" << std::endl;
    std::cout << std::endl;

    return 0;
}
