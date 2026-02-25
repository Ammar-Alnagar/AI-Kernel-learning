/**
 * @file 02_shape_and_stride.cu
 * @brief Understanding Shape and Stride in CuTe Layouts
 * 
 * This tutorial dives deeper into:
 * - How shape defines the logical dimensions
 * - How stride controls memory traversal
 * - Custom strides for padded layouts
 * - Static vs dynamic shapes
 */

#include <cute/layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== CuTe Tutorial: Shape and Stride ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: Shape Defines Logical Dimensions
    // ================================================================
    std::cout << "--- Concept 1: Shape ---" << std::endl;
    std::cout << "Shape defines the size of each dimension in logical space." << std::endl;
    std::cout << std::endl;

    // 1D shape (vector)
    auto shape_1d = make_shape(Int<8>{});
    std::cout << "1D Shape (vector of 8): ";
    print(shape_1d);
    std::cout << std::endl;

    // 2D shape (matrix)
    auto shape_2d = make_shape(Int<4>{}, Int<6>{});
    std::cout << "2D Shape (4x6 matrix): ";
    print(shape_2d);
    std::cout << std::endl;

    // 3D shape (tensor)
    auto shape_3d = make_shape(Int<2>{}, Int<3>{}, Int<4>{});
    std::cout << "3D Shape (2x3x4 tensor): ";
    print(shape_3d);
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 2: Stride Controls Memory Traversal
    // ================================================================
    std::cout << "--- Concept 2: Stride ---" << std::endl;
    std::cout << "Stride defines how many memory elements to skip for each dimension." << std::endl;
    std::cout << std::endl;

    // For a shape [M, N], stride [S_M, S_N] means:
    // offset = i * S_M + j * S_N
    
    std::cout << "For shape [M, N] with stride [S_M, S_N]:" << std::endl;
    std::cout << "  offset(i, j) = i * S_M + j * S_N" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Row-Major Stride Derivation
    // ================================================================
    std::cout << "--- Example 1: Row-Major Stride ---" << std::endl;
    
    // For a 4x6 row-major matrix:
    // - To move to next row (i+1), skip 6 elements (the row width)
    // - To move to next column (j+1), skip 1 element
    // Therefore: stride = [6, 1]
    
    auto row_major = make_layout(
        make_shape(Int<4>{}, Int<6>{}),
        make_stride(Int<6>{}, Int<1>{})
    );
    
    std::cout << "4x6 Row-Major Layout: ";
    print(row_major);
    std::cout << std::endl;
    std::cout << "Stride: [6, 1] - row stride = column count, col stride = 1" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Column-Major Stride Derivation
    // ================================================================
    std::cout << "--- Example 2: Column-Major Stride ---" << std::endl;
    
    // For a 4x6 column-major matrix:
    // - To move to next row (i+1), skip 1 element
    // - To move to next column (j+1), skip 4 elements (the column height)
    // Therefore: stride = [1, 4]
    
    auto col_major = make_layout(
        make_shape(Int<4>{}, Int<6>{}),
        make_stride(Int<1>{}, Int<4>{})
    );
    
    std::cout << "4x6 Column-Major Layout: ";
    print(col_major);
    std::cout << std::endl;
    std::cout << "Stride: [1, 4] - row stride = 1, col stride = row count" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Padded Layout (Custom Stride)
    // ================================================================
    std::cout << "--- Example 3: Padded Layout ---" << std::endl;
    std::cout << "Padding is useful for memory alignment and avoiding bank conflicts." << std::endl;
    std::cout << std::endl;
    
    // Create a 4x6 layout with padding
    // Actual row stride = 8 (padding of 2 elements per row)
    auto padded_layout = make_layout(
        make_shape(Int<4>{}, Int<6>{}),
        make_stride(Int<8>{}, Int<1>{})  // Row stride = 8 instead of 6
    );
    
    std::cout << "4x6 Padded Layout (stride=[8,1]): ";
    print(padded_layout);
    std::cout << std::endl;
    
    std::cout << "Visualizing the padding (X = valid, . = padding):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("Row %d: ", i);
        for (int j = 0; j < 8; ++j) {
            if (j < 6) {
                printf("%3d ", padded_layout(i, j));
            } else {
                printf("  X ");  // Padding elements
            }
        }
        std::cout << std::endl;
    }
    std::cout << "Note: Padding elements (X) are never accessed logically." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Comparing Memory Offsets
    // ================================================================
    std::cout << "--- Example 4: Comparing Layouts ---" << std::endl;
    std::cout << "Same logical position (2, 3) in different layouts:" << std::endl;
    std::cout << std::endl;
    
    int row = 2, col = 3;
    
    int row_major_offset = row_major(row, col);
    int col_major_offset = col_major(row, col);
    int padded_offset = padded_layout(row, col);
    
    std::cout << "  Position (2, 3):" << std::endl;
    std::cout << "    Row-major offset:    " << row_major_offset << std::endl;
    std::cout << "    Column-major offset: " << col_major_offset << std::endl;
    std::cout << "    Padded offset:       " << padded_offset << std::endl;
    std::cout << std::endl;
    
    std::cout << "Verification:" << std::endl;
    std::cout << "  Row-major:    2*6 + 3*1 = " << (2*6 + 3*1) << std::endl;
    std::cout << "  Column-major: 2*1 + 3*4 = " << (2*1 + 3*4) << std::endl;
    std::cout << "  Padded:       2*8 + 3*1 = " << (2*8 + 3*1) << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Static vs Dynamic Shapes
    // ================================================================
    std::cout << "--- Example 5: Static vs Dynamic Shapes ---" << std::endl;
    std::cout << "CuTe supports both compile-time (static) and runtime (dynamic) sizes." << std::endl;
    std::cout << std::endl;
    
    // Static shape (known at compile time)
    auto static_shape = make_shape(Int<4>{}, Int<6>{});
    std::cout << "Static shape (Int<4>, Int<6>): ";
    print(static_shape);
    std::cout << "  <- Known at compile time" << std::endl;
    
    // Dynamic shape (known at runtime)
    int M = 4, N = 6;
    auto dynamic_shape = make_shape(M, N);
    std::cout << "Dynamic shape (M=4, N=6):    ";
    print(dynamic_shape);
    std::cout << "  <- Known at runtime" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Static shapes enable better compiler optimization." << std::endl;
    std::cout << "Dynamic shapes provide flexibility for runtime sizes." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Shape defines logical dimensions (M, N, ...)" << std::endl;
    std::cout << "2. Stride defines memory step sizes [S_M, S_N, ...]" << std::endl;
    std::cout << "3. Row-major: stride = [N, 1] for MxN matrix" << std::endl;
    std::cout << "4. Column-major: stride = [1, M] for MxN matrix" << std::endl;
    std::cout << "5. Custom strides enable padding and special layouts" << std::endl;
    std::cout << "6. offset(i,j) = i*S_M + j*S_N" << std::endl;
    std::cout << "7. Use Int<N>{} for static, variables for dynamic shapes" << std::endl;
    std::cout << std::endl;

    return 0;
}
