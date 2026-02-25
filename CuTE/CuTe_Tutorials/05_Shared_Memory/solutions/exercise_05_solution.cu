/**
 * @file exercise_05_solution.cu
 * @brief Solution: Shared Memory and Swizzling
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== Exercise 05: Solution ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 1: Padding for 16x16 Tile
    // ================================================================
    std::cout << "--- Exercise 1: Padding for 16x16 Tile ---" << std::endl;
    
    const int TILE_M = 16;
    const int TILE_K = 16;
    const int NUM_BANKS = 32;
    
    // For 32 banks, we want consecutive columns to map to different banks
    // With row-major, column access has stride = row_stride
    // We want row_stride % 32 != 0 to avoid all threads hitting same bank
    
    // Solution: Add padding to make stride = 17 (or any odd number not divisible by 32)
    const int PADDED_STRIDE = 17;  // TILE_K + 1
    
    auto padded_layout = make_layout(
        make_shape(Int<TILE_M>{}, Int<TILE_K>{}),
        make_stride(Int<PADDED_STRIDE>{}, Int<1>{})
    );
    
    std::cout << "16x16 tile with 32 banks:" << std::endl;
    std::cout << "  Original stride: 16" << std::endl;
    std::cout << "  Padded stride: " << PADDED_STRIDE << std::endl;
    std::cout << std::endl;
    
    std::cout << "Column 0 access with padded layout:" << std::endl;
    for (int row = 0; row < TILE_M; ++row) {
        int offset = padded_layout(row, 0);
        int bank = offset % NUM_BANKS;
        printf("  Row %2d: offset=%3d, bank=%2d\n", row, offset, bank);
    }
    std::cout << std::endl;
    
    std::cout << "All 16 threads access different banks - no conflicts!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 2: GEMM Shared Memory Size
    // ================================================================
    std::cout << "--- Exercise 2: GEMM Shared Memory Size ---" << std::endl;
    
    const int GEMM_M = 32;
    const int GEMM_N = 32;
    const int GEMM_K = 32;
    const int PADDING = 1;
    
    int smem_A = GEMM_M * (GEMM_K + PADDING);  // 32 * 33 = 1056
    int smem_B = (GEMM_K + PADDING) * GEMM_N;  // 33 * 32 = 1056
    int total_elements = smem_A + smem_B;       // 2112
    int total_bytes = total_elements * sizeof(float);  // 8448 bytes
    
    std::cout << "32x32x32 GEMM tile with padding:" << std::endl;
    std::cout << "  A tile: " << GEMM_M << "x" << (GEMM_K + PADDING) 
              << " = " << smem_A << " elements" << std::endl;
    std::cout << "  B tile: " << (GEMM_K + PADDING) << "x" << GEMM_N 
              << " = " << smem_B << " elements" << std::endl;
    std::cout << "  Total elements: " << total_elements << std::endl;
    std::cout << "  Total bytes: " << total_bytes << " (" << (total_bytes / 1024.0) << " KB)" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 3: XOR Swizzle Pattern
    // ================================================================
    std::cout << "--- Exercise 3: XOR Swizzle Pattern ---" << std::endl;
    
    std::cout << "For a 16x16 tile, a good XOR swizzle pattern is:" << std::endl;
    std::cout << "  swizzle = (row XOR col) << 3" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Explanation:" << std::endl;
    std::cout << "  - XOR row and col to mix the coordinates" << std::endl;
    std::cout << "  - Shift by 3 to affect higher-order bits" << std::endl;
    std::cout << "  - This spreads accesses across 8+ banks" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Example for 4x4 subset:" << std::endl;
    for (int row = 0; row < 4; ++row) {
        printf("  Row %d: ", row);
        for (int col = 0; col < 4; ++col) {
            int base = row * 4 + col;
            int swizzle = ((row ^ col) << 3) & 0x1F;
            int swizzled = base ^ swizzle;
            printf("%2d ", swizzled);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Exercise 1: Padded stride = 17 for 16x16 tile with 32 banks" << std::endl;
    std::cout << "Exercise 2: 32x32x32 GEMM needs 8448 bytes shared memory" << std::endl;
    std::cout << "Exercise 3: swizzle = (row XOR col) << 3 for 16x16 tile" << std::endl;
    std::cout << std::endl;

    return 0;
}
