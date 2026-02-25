/**
 * @file exercise_04.cu
 * @brief Exercise: MMA Atoms
 * 
 * Exercise 1: For a 16x16x16 MMA shape, calculate:
 *   - Number of elements in A, B, and D matrices
 *   - Total FLOPs per MMA operation
 * 
 * Exercise 2: For a 64x64x64 GEMM with 16x16x16 tiles:
 *   - How many MMA operations are needed?
 *   - How many K-tiles must be accumulated?
 * 
 * Exercise 3: Write the accumulation formula for a GEMM with 4 K-tiles.
 * 
 * Instructions: Complete the calculations and verify with the solution.
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== Exercise 04: MMA Atoms ===" << std::endl;
    std::cout << std::endl;

    // Exercise 1: MMA shape calculations
    std::cout << "--- Exercise 1: 16x16x16 MMA Shape ---" << std::endl;
    std::cout << "TODO: Calculate elements in A, B, D and total FLOPs" << std::endl;
    std::cout << "  M = 16, N = 16, K = 16" << std::endl;
    std::cout << "  Elements in A (MxK): ?" << std::endl;
    std::cout << "  Elements in B (KxN): ?" << std::endl;
    std::cout << "  Elements in D (MxN): ?" << std::endl;
    std::cout << "  FLOPs (2*M*N*K): ?" << std::endl;
    std::cout << std::endl;

    // Exercise 2: Tiling calculations
    std::cout << "--- Exercise 2: 64x64x64 GEMM Tiling ---" << std::endl;
    std::cout << "TODO: Calculate number of MMA operations and K-tiles" << std::endl;
    std::cout << "  GEMM: 64x64x64, Tile: 16x16x16" << std::endl;
    std::cout << "  Tiles along M: ?" << std::endl;
    std::cout << "  Tiles along N: ?" << std::endl;
    std::cout << "  Tiles along K: ?" << std::endl;
    std::cout << "  Total MMA operations: ?" << std::endl;
    std::cout << std::endl;

    // Exercise 3: Accumulation formula
    std::cout << "--- Exercise 3: Accumulation Formula ---" << std::endl;
    std::cout << "TODO: Write the formula for D with 4 K-tiles" << std::endl;
    std::cout << "  D = ?" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "See exercise_04_solution.cu for answers" << std::endl;

    return 0;
}
