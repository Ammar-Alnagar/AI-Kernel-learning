/**
 * @file exercise_04_solution.cu
 * @brief Solution: MMA Atoms
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== Exercise 04: Solution ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 1: 16x16x16 MMA Shape
    // ================================================================
    std::cout << "--- Exercise 1: 16x16x16 MMA Shape ---" << std::endl;
    
    const int M = 16, N = 16, K = 16;
    
    int elements_A = M * K;
    int elements_B = K * N;
    int elements_D = M * N;
    int total_elements = elements_A + elements_B + elements_D;
    int flops = 2 * M * N * K;
    
    std::cout << "MMA Shape: " << M << "x" << N << "x" << K << std::endl;
    std::cout << std::endl;
    std::cout << "Elements in A (MxK = 16x16): " << elements_A << std::endl;
    std::cout << "Elements in B (KxN = 16x16): " << elements_B << std::endl;
    std::cout << "Elements in D (MxN = 16x16): " << elements_D << std::endl;
    std::cout << "Total elements: " << total_elements << std::endl;
    std::cout << "Total FLOPs (2*M*N*K): " << flops << std::endl;
    std::cout << std::endl;
    std::cout << "Verification:" << std::endl;
    std::cout << "  A: 16*16 = 256 ✓" << std::endl;
    std::cout << "  B: 16*16 = 256 ✓" << std::endl;
    std::cout << "  D: 16*16 = 256 ✓" << std::endl;
    std::cout << "  FLOPs: 2*16*16*16 = 8192 ✓" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 2: 64x64x64 GEMM Tiling
    // ================================================================
    std::cout << "--- Exercise 2: 64x64x64 GEMM Tiling ---" << std::endl;
    
    const int GEMM_M = 64, GEMM_N = 64, GEMM_K = 64;
    const int TILE_M = 16, TILE_N = 16, TILE_K = 16;
    
    int tiles_M = (GEMM_M + TILE_M - 1) / TILE_M;
    int tiles_N = (GEMM_N + TILE_N - 1) / TILE_N;
    int tiles_K = (GEMM_K + TILE_K - 1) / TILE_K;
    int total_mma_ops = tiles_M * tiles_N * tiles_K;
    
    std::cout << "GEMM: " << GEMM_M << "x" << GEMM_N << "x" << GEMM_K << std::endl;
    std::cout << "Tile: " << TILE_M << "x" << TILE_N << "x" << TILE_K << std::endl;
    std::cout << std::endl;
    std::cout << "Tiles along M: " << tiles_M << " (64/16 = 4)" << std::endl;
    std::cout << "Tiles along N: " << tiles_N << " (64/16 = 4)" << std::endl;
    std::cout << "Tiles along K: " << tiles_K << " (64/16 = 4)" << std::endl;
    std::cout << std::endl;
    std::cout << "Total MMA operations: " << total_mma_ops << std::endl;
    std::cout << "  (= tiles_M * tiles_N * tiles_K = 4 * 4 * 4 = 64)" << std::endl;
    std::cout << std::endl;
    std::cout << "K-tiles to accumulate per output tile: " << tiles_K << " (4)" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 3: Accumulation Formula
    // ================================================================
    std::cout << "--- Exercise 3: Accumulation Formula ---" << std::endl;
    
    std::cout << "For GEMM with 4 K-tiles, the accumulation formula is:" << std::endl;
    std::cout << std::endl;
    std::cout << "  D = A₀*B₀ + A₁*B₁ + A₂*B₂ + A₃*B₃" << std::endl;
    std::cout << std::endl;
    std::cout << "Or in summation notation:" << std::endl;
    std::cout << "  D = Σ(k=0 to 3) A_k * B_k" << std::endl;
    std::cout << std::endl;
    std::cout << "Expanded:" << std::endl;
    std::cout << "  Initialize: D = 0" << std::endl;
    std::cout << "  Iteration 0: D += A₀ * B₀" << std::endl;
    std::cout << "  Iteration 1: D += A₁ * B₁" << std::endl;
    std::cout << "  Iteration 2: D += A₂ * B₂" << std::endl;
    std::cout << "  Iteration 3: D += A₃ * B₃" << std::endl;
    std::cout << "  Final: D = A₀*B₀ + A₁*B₁ + A₂*B₂ + A₃*B₃" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Exercise 1: 16x16x16 MMA has 256 elements per operand, 8192 FLOPs" << std::endl;
    std::cout << "Exercise 2: 64x64x64 GEMM needs 64 MMA operations (4x4x4 grid)" << std::endl;
    std::cout << "Exercise 3: D = Σ(k=0 to 3) A_k * B_k (accumulate 4 K-tiles)" << std::endl;
    std::cout << std::endl;

    return 0;
}
