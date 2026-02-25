/**
 * @file 02_swizzling.cu
 * @brief Swizzling for Bank Conflict Avoidance
 * 
 * This tutorial covers swizzling techniques:
 * - What is swizzling
 * - How swizzling avoids bank conflicts
 * - Implementing swizzled layouts in CuTe
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
    std::cout << "=== CuTe Tutorial: Swizzling ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: What is Swizzling?
    // ================================================================
    std::cout << "--- Concept 1: What is Swizzling? ---" << std::endl;
    std::cout << "Swizzling is a technique to XOR bits of the address to spread" << std::endl;
    std::cout << "memory accesses across banks and avoid conflicts." << std::endl;
    std::cout << std::endl;
    std::cout << "Basic idea:" << std::endl;
    std::cout << "  swizzled_offset = offset XOR swizzle_pattern" << std::endl;
    std::cout << std::endl;
    std::cout << "This scrambles the address mapping to distribute accesses." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Simple XOR Swizzle
    // ================================================================
    std::cout << "--- Example 1: Simple XOR Swizzle ---" << std::endl;
    
    const int NUM_ELEMENTS = 32;
    const int SWIZZLE_BITS = 3;  // XOR with bit 3
    
    std::cout << "XOR swizzle with bit position " << SWIZZLE_BITS << ":" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Original -> Swizzled mapping:" << std::endl;
    for (int i = 0; i < NUM_ELEMENTS; ++i) {
        int swizzled = i ^ (1 << SWIZZLE_BITS);  // XOR with bit 3
        printf("  %2d (0x%02X) -> %2d (0x%02X)\n", i, i, swizzled, swizzled);
    }
    std::cout << std::endl;
    
    std::cout << "Notice how addresses are permuted!" << std::endl;
    std::cout << "This helps distribute consecutive accesses across banks." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Bank Conflict Without Swizzling
    // ================================================================
    std::cout << "--- Example 2: Bank Conflict Without Swizzling ---" << std::endl;
    
    // 8x8 row-major layout
    auto normal_layout = make_layout(make_shape(Int<8>{}, Int<8>{}), GenRowMajor{});
    
    std::cout << "8x8 Row-Major Layout - Column Access Pattern:" << std::endl;
    std::cout << "When threads access the same column (vertical stride):" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Column 0:" << std::endl;
    for (int row = 0; row < 8; ++row) {
        int offset = normal_layout(row, 0);
        int bank = offset % 8;  // Simplified: 8 banks for demonstration
        printf("  Thread %d: offset=%2d, bank=%d\n", row, offset, bank);
    }
    std::cout << std::endl;
    
    std::cout << "Column 1:" << std::endl;
    for (int row = 0; row < 8; ++row) {
        int offset = normal_layout(row, 1);
        int bank = offset % 8;
        printf("  Thread %d: offset=%2d, bank=%d\n", row, offset, bank);
    }
    std::cout << std::endl;
    
    std::cout << "Problem: All threads access different banks (good for this case)!" << std::endl;
    std::cout << "But with different configurations, conflicts can occur." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Swizzled Layout
    // ================================================================
    std::cout << "--- Example 3: Swizzled Layout ---" << std::endl;
    
    // Create a simple swizzled layout manually
    // In practice, CuTe provides swizzle layout generators
    
    std::cout << "Swizzled address calculation:" << std::endl;
    std::cout << "  swizzled_offset = base_offset XOR (row XOR column) << shift" << std::endl;
    std::cout << std::endl;
    
    // Demonstrate XOR-based swizzling
    const int SWIZZLE_SHIFT = 3;
    
    std::cout << "8x8 layout with XOR swizzle (shift=" << SWIZZLE_SHIFT << "):" << std::endl;
    std::cout << std::endl;
    
    for (int row = 0; row < 8; ++row) {
        printf("Row %d: ", row);
        for (int col = 0; col < 8; ++col) {
            int base_offset = row * 8 + col;
            int swizzle_factor = (row ^ col) << SWIZZLE_SHIFT;
            int swizzled = base_offset ^ (swizzle_factor & 0x1F);  // Keep in range
            printf("%2d ", swizzled);
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 4: Common Swizzle Patterns
    // ================================================================
    std::cout << "--- Example 4: Common Swizzle Patterns ---" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Pattern 1: XOR row and column bits" << std::endl;
    std::cout << "  swizzle = (row XOR col) << shift" << std::endl;
    std::cout << "  Good for 2D access patterns" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Pattern 2: XOR with thread ID bits" << std::endl;
    std::cout << "  swizzle = (tid >> shift) & mask" << std::endl;
    std::cout << "  Good for thread-to-data mapping" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Pattern 3: Combined swizzle" << std::endl;
    std::cout << "  swizzle = ((row XOR col) << s1) XOR (tid >> s2)" << std::endl;
    std::cout << "  Maximum bank conflict avoidance" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Swizzle Effect on Bank Distribution
    // ================================================================
    std::cout << "--- Example 5: Bank Distribution Analysis ---" << std::endl;
    
    const int NUM_BANKS = 8;  // Simplified for demonstration
    const int TILE_SIZE = 64;
    
    // Count bank accesses without swizzle
    std::vector<int> bank_count_no_swizzle(NUM_BANKS, 0);
    std::vector<int> bank_count_with_swizzle(NUM_BANKS, 0);
    
    for (int i = 0; i < TILE_SIZE; ++i) {
        // Without swizzle
        int bank_normal = i % NUM_BANKS;
        bank_count_no_swizzle[bank_normal]++;
        
        // With swizzle (XOR with bit 3)
        int swizzled = i ^ (i >> 3);
        int bank_swizzle = swizzled % NUM_BANKS;
        bank_count_with_swizzle[bank_swizzle]++;
    }
    
    std::cout << "Bank access distribution (64 accesses, 8 banks):" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Without swizzle:" << std::endl;
    for (int b = 0; b < NUM_BANKS; ++b) {
        printf("  Bank %d: %d accesses\n", b, bank_count_no_swizzle[b]);
    }
    std::cout << std::endl;
    
    std::cout << "With swizzle:" << std::endl;
    for (int b = 0; b < NUM_BANKS; ++b) {
        printf("  Bank %d: %d accesses\n", b, bank_count_with_swizzle[b]);
    }
    std::cout << std::endl;
    
    std::cout << "Good swizzling distributes accesses evenly across banks!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: Swizzle in GEMM Context
    // ================================================================
    std::cout << "--- Example 6: Swizzle in GEMM ---" << std::endl;
    std::cout << "In GEMM, shared memory tiles are accessed by multiple threads." << std::endl;
    std::cout << "Swizzling ensures even the worst-case access patterns are safe." << std::endl;
    std::cout << std::endl;
    
    std::cout << "Typical GEMM shared memory layout:" << std::endl;
    std::cout << "  - Matrix A tile: M x K" << std::endl;
    std::cout << "  - Matrix B tile: K x N" << std::endl;
    std::cout << "  - Swizzle applied to both tiles" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Benefits:" << std::endl;
    std::cout << "  - No bank conflicts regardless of access pattern" << std::endl;
    std::cout << "  - Consistent performance" << std::endl;
    std::cout << "  - Essential for high-performance GEMM" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Swizzling XORs address bits to spread accesses" << std::endl;
    std::cout << "2. Prevents bank conflicts in shared memory" << std::endl;
    std::cout << "3. Common pattern: XOR row and column bits" << std::endl;
    std::cout << "4. Provides consistent performance" << std::endl;
    std::cout << "5. Essential for high-performance GEMM kernels" << std::endl;
    std::cout << "6. CuTe provides swizzle layout generators" << std::endl;
    std::cout << std::endl;

    return 0;
}
