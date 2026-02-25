/**
 * @file exercise_05.cu
 * @brief Exercise: Shared Memory and Swizzling
 * 
 * Exercise 1: For a 16x16 shared memory tile with 32 banks:
 *   - Calculate the padding needed to avoid bank conflicts
 *   - What is the new stride?
 * 
 * Exercise 2: Calculate the shared memory size for a 32x32x32 GEMM tile
 *   with padding of 1 element per row.
 * 
 * Exercise 3: What XOR swizzle pattern would you use for a 16x16 tile?
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
    std::cout << "=== Exercise 05: Shared Memory ===" << std::endl;
    std::cout << std::endl;

    // Exercise 1: Padding calculation
    std::cout << "--- Exercise 1: Padding for 16x16 Tile ---" << std::endl;
    std::cout << "TODO: Calculate padding for 32 banks" << std::endl;
    std::cout << "  Original stride: 16" << std::endl;
    std::cout << "  Padded stride: ?" << std::endl;
    std::cout << std::endl;

    // Exercise 2: Shared memory size
    std::cout << "--- Exercise 2: GEMM Shared Memory Size ---" << std::endl;
    std::cout << "TODO: Calculate for 32x32x32 GEMM with padding" << std::endl;
    std::cout << "  A tile size: ?" << std::endl;
    std::cout << "  B tile size: ?" << std::endl;
    std::cout << "  Total bytes: ?" << std::endl;
    std::cout << std::endl;

    // Exercise 3: XOR swizzle pattern
    std::cout << "--- Exercise 3: XOR Swizzle Pattern ---" << std::endl;
    std::cout << "TODO: Suggest an XOR pattern for 16x16 tile" << std::endl;
    std::cout << "  swizzle = (row XOR col) << ?" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "See exercise_05_solution.cu for answers" << std::endl;

    return 0;
}
