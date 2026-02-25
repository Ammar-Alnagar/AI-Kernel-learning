/**
 * @file 02_mma_shapes.cu
 * @brief MMA Shapes and Configurations
 * 
 * This tutorial covers different MMA shapes and their use cases:
 * - Standard MMA shapes
 * - Shape selection for different problems
 * - Thread block tiling for MMA
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== CuTe Tutorial: MMA Shapes ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: MMA Shape Notation
    // ================================================================
    std::cout << "--- Concept 1: MMA Shape Notation ---" << std::endl;
    std::cout << "MMA shape is described as (M, N, K) where:" << std::endl;
    std::cout << "  M = number of rows in output matrix D" << std::endl;
    std::cout << "  N = number of columns in output matrix D" << std::endl;
    std::cout << "  K = reduction dimension (inner dimension)" << std::endl;
    std::cout << std::endl;
    std::cout << "For D[MxN] = A[MxK] * B[KxN]:" << std::endl;
    std::cout << "  Each MMA atom computes a tile of this operation" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Common MMA Shapes
    // ================================================================
    std::cout << "--- Example 1: Common MMA Shapes ---" << std::endl;
    std::cout << std::endl;
    
    struct MMAShape {
        int M, N, K;
        const char* name;
        const char* description;
    };
    
    std::vector<MMAShape> mma_shapes = {
        {16, 8, 16, "16x8x16", "Narrow output, good for GEMV-like patterns"},
        {16, 16, 16, "16x16x16", "Square output, balanced computation"},
        {32, 8, 16, "32x8x16", "Tall output, good for attention Q*K^T"},
        {8, 32, 16, "8x32x16", "Wide output, good for projection layers"},
    };
    
    std::cout << "Common Tensor Core MMA Shapes (Volta/Ampere):" << std::endl;
    std::cout << std::endl;
    
    for (const auto& shape : mma_shapes) {
        std::cout << "  " << shape.name << ":" << std::endl;
        std::cout << "    " << shape.description << std::endl;
        std::cout << "    Elements in D: " << (shape.M * shape.N) << std::endl;
        std::cout << "    FLOPs per MMA: " << (2 * shape.M * shape.N * shape.K) << std::endl;
        std::cout << std::endl;
    }

    // ================================================================
    // Example 2: MMA Layout for 16x16x16
    // ================================================================
    std::cout << "--- Example 2: 16x16x16 MMA Layout ---" << std::endl;
    
    const int MMA_M = 16, MMA_N = 16, MMA_K = 16;
    
    // Layouts for MMA operands
    auto layout_A = make_layout(make_shape(Int<MMA_M>{}, Int<MMA_K>{}), GenRowMajor{});
    auto layout_B = make_layout(make_shape(Int<MMA_K>{}, Int<MMA_N>{}), GenColMajor{});
    auto layout_D = make_layout(make_shape(Int<MMA_M>{}, Int<MMA_N>{}), GenRowMajor{});
    
    std::cout << "MMA Shape: " << MMA_M << "x" << MMA_N << "x" << MMA_K << std::endl;
    std::cout << std::endl;
    
    std::cout << "Operand A layout (" << MMA_M << "x" << MMA_K << "):" << std::endl;
    std::cout << "  Shape:  "; print(layout_A.shape()); std::cout << std::endl;
    std::cout << "  Stride: "; print(layout_A.stride()); std::cout << std::endl;
    std::cout << std::endl;
    
    std::cout << "Operand B layout (" << MMA_K << "x" << MMA_N << "):" << std::endl;
    std::cout << "  Shape:  "; print(layout_B.shape()); std::cout << std::endl;
    std::cout << "  Stride: "; print(layout_B.stride()); std::cout << std::endl;
    std::cout << std::endl;
    
    std::cout << "Output D layout (" << MMA_M << "x" << MMA_N << "):" << std::endl;
    std::cout << "  Shape:  "; print(layout_D.shape()); std::cout << std::endl;
    std::cout << "  Stride: "; print(layout_D.stride()); std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Element Count Analysis
    // ================================================================
    std::cout << "--- Example 3: Element Count Analysis ---" << std::endl;
    std::cout << std::endl;
    
    for (const auto& shape : mma_shapes) {
        int elements_A = shape.M * shape.K;
        int elements_B = shape.K * shape.N;
        int elements_D = shape.M * shape.N;
        int total_elements = elements_A + elements_B + elements_D;
        int flops = 2 * shape.M * shape.N * shape.K;
        
        std::cout << shape.name << ":" << std::endl;
        std::cout << "  A: " << elements_A << " elements (" << shape.M << "x" << shape.K << ")" << std::endl;
        std::cout << "  B: " << elements_B << " elements (" << shape.K << "x" << shape.N << ")" << std::endl;
        std::cout << "  D: " << elements_D << " elements (" << shape.M << "x" << shape.N << ")" << std::endl;
        std::cout << "  Total: " << total_elements << " elements" << std::endl;
        std::cout << "  Compute: " << flops << " FLOPs" << std::endl;
        std::cout << "  Efficiency: " << (flops / static_cast<float>(total_elements)) 
                  << " FLOPs/element" << std::endl;
        std::cout << std::endl;
    }

    // ================================================================
    // Example 4: Tiling a Large GEMM
    // ================================================================
    std::cout << "--- Example 4: Tiling a Large GEMM ---" << std::endl;
    
    const int GEMM_M = 128, GEMM_N = 128, GEMM_K = 256;
    const int TILE_M = 16, TILE_N = 16, TILE_K = 16;
    
    int tiles_M = (GEMM_M + TILE_M - 1) / TILE_M;
    int tiles_N = (GEMM_N + TILE_N - 1) / TILE_N;
    int tiles_K = (GEMM_K + TILE_K - 1) / TILE_K;
    
    std::cout << "Large GEMM: D[" << GEMM_M << "x" << GEMM_N << "] = A[" 
              << GEMM_M << "x" << GEMM_K << "] * B[" << GEMM_K << "x" << GEMM_N << "]" << std::endl;
    std::cout << "MMA Tile: " << TILE_M << "x" << TILE_N << "x" << TILE_K << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tiling:" << std::endl;
    std::cout << "  Tiles along M: " << tiles_M << std::endl;
    std::cout << "  Tiles along N: " << tiles_N << std::endl;
    std::cout << "  Tiles along K: " << tiles_K << std::endl;
    std::cout << std::endl;
    
    std::cout << "Total MMA operations: " << (tiles_M * tiles_N * tiles_K) << std::endl;
    std::cout << std::endl;
    
    std::cout << "Visualization (M-N plane):" << std::endl;
    std::cout << "  " << std::string(tiles_N * 5, '-') << std::endl;
    for (int i = 0; i < tiles_M; ++i) {
        printf("  |");
        for (int j = 0; j < tiles_N; ++j) {
            printf(" T%2d", i * tiles_N + j);
        }
        printf(" |\n");
    }
    std::cout << "  " << std::string(tiles_N * 5, '-') << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Shape Selection Guidelines
    // ================================================================
    std::cout << "--- Example 5: Shape Selection Guidelines ---" << std::endl;
    std::cout << std::endl;
    
    std::cout << "When to use each shape:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "16x8x16:" << std::endl;
    std::cout << "  - Good for matrix-vector multiply (GEMV)" << std::endl;
    std::cout << "  - When N dimension is small" << std::endl;
    std::cout << "  - Lower register pressure" << std::endl;
    std::cout << std::endl;
    
    std::cout << "16x16x16:" << std::endl;
    std::cout << "  - Balanced compute and memory" << std::endl;
    std::cout << "  - Good default choice" << std::endl;
    std::cout << "  - Square output tiles" << std::endl;
    std::cout << std::endl;
    
    std::cout << "32x8x16 or 8x32x16:" << std::endl;
    std::cout << "  - When one dimension is much larger" << std::endl;
    std::cout << "  - Attention mechanisms (Q*K^T)" << std::endl;
    std::cout << "  - Better occupancy in some cases" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. MMA shape (M, N, K) defines the tile size" << std::endl;
    std::cout << "2. Common shapes: 16x8x16, 16x16x16, 32x8x16, 8x32x16" << std::endl;
    std::cout << "3. 16x16x16 is a good balanced default" << std::endl;
    std::cout << "4. Large GEMMs are tiled in all 3 dimensions" << std::endl;
    std::cout << "5. Shape choice affects register pressure and occupancy" << std::endl;
    std::cout << "6. Match shape to problem characteristics" << std::endl;
    std::cout << std::endl;

    return 0;
}
