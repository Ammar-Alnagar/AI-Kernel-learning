/**
 * @file exercise_03.cu
 * @brief Exercise: Tiled Copy
 * 
 * Exercise 1: Create a 2D copy kernel for an 8x8 matrix using a 4x4 thread block.
 * 
 * Exercise 2: Modify the kernel so each thread copies 2 elements instead of 1.
 * 
 * Exercise 3: Calculate the thread ID to 2D coordinate mapping for a 4x4 thread block.
 * 
 * Instructions: Fill in the TODO sections and verify with the solution.
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
    std::cout << "=== Exercise 03: Tiled Copy ===" << std::endl;
    std::cout << std::endl;

    // Exercise 1: TODO - Create 8x8 copy with 4x4 thread block
    std::cout << "--- Exercise 1: 8x8 Copy with 4x4 Block ---" << std::endl;
    std::cout << "TODO: Create thread layout for 4x4 block" << std::endl;
    std::cout << "TODO: Launch kernel where each thread copies one element" << std::endl;
    std::cout << std::endl;

    // Exercise 2: TODO - Each thread copies 2 elements
    std::cout << "--- Exercise 2: Vectorized Copy ---" << std::endl;
    std::cout << "TODO: Modify so each thread copies 2 consecutive elements" << std::endl;
    std::cout << std::endl;

    // Exercise 3: TODO - Calculate thread mapping
    std::cout << "--- Exercise 3: Thread Mapping ---" << std::endl;
    std::cout << "TODO: For a 4x4 thread block with row-major layout," << std::endl;
    std::cout << "      what 2D coordinates does thread ID 5 map to?" << std::endl;
    std::cout << "Expected: (1, 1) because 5 = 1*4 + 1" << std::endl;
    std::cout << std::endl;

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "See exercise_03_solution.cu for complete solution" << std::endl;

    return 0;
}
