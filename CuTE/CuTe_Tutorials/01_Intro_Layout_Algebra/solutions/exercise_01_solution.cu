/**
 * @file exercise_01_solution.cu
 * @brief Solution: Practice Layout Algebra
 * 
 * This file contains the complete solution for Exercise 01.
 */

#include <cute/layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== Exercise 01: Solution ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 1: Create a 6x8 row-major layout
    // ================================================================
    std::cout << "--- Exercise 1: 6x8 Row-Major Layout ---" << std::endl;
    
    auto layout_ex1 = make_layout(
        make_shape(Int<6>{}, Int<8>{}),
        GenRowMajor{}
    );
    
    std::cout << "Layout: ";
    print(layout_ex1);
    std::cout << std::endl;
    
    std::cout << "Shape:  ";
    print(layout_ex1.shape());
    std::cout << std::endl;
    
    std::cout << "Stride: ";
    print(layout_ex1.stride());
    std::cout << std::endl;
    
    std::cout << "Verification:" << std::endl;
    std::cout << "  Shape matches expected (6, 8): " 
              << (get<0>(layout_ex1.shape()) == 6 && get<1>(layout_ex1.shape()) == 8 ? "YES" : "NO") 
              << std::endl;
    std::cout << "  Stride matches expected (8, 1): " 
              << (get<0>(layout_ex1.stride()) == 8 && get<1>(layout_ex1.stride()) == 1 ? "YES" : "NO") 
              << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 2: Create a padded 5x5 layout with row stride 8
    // ================================================================
    std::cout << "--- Exercise 2: Padded 5x5 Layout ---" << std::endl;
    
    auto padded_layout = make_layout(
        make_shape(Int<5>{}, Int<5>{}),
        make_stride(Int<8>{}, Int<1>{})
    );
    
    std::cout << "Padded Layout: ";
    print(padded_layout);
    std::cout << std::endl;
    
    int offset_ex2 = padded_layout(3, 4);
    std::cout << "Offset at position (3, 4): " << offset_ex2 << std::endl;
    
    std::cout << "Verification:" << std::endl;
    std::cout << "  Expected: 3*8 + 4*1 = 28" << std::endl;
    std::cout << "  Got:      " << offset_ex2 << std::endl;
    std::cout << "  Match: " << (offset_ex2 == 28 ? "YES" : "NO") << std::endl;
    
    // Visualize the padded layout
    std::cout << "\nVisual layout (numbers = offsets, X = padding):" << std::endl;
    for (int i = 0; i < 5; ++i) {
        printf("Row %d: ", i);
        for (int j = 0; j < 8; ++j) {
            if (j < 5) {
                printf("%2d  ", padded_layout(i, j));
            } else {
                printf(" X  ");
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Exercise 3: Create a 4x6 column-major layout
    // ================================================================
    std::cout << "--- Exercise 3: 4x6 Column-Major Layout ---" << std::endl;
    
    auto col_layout = make_layout(
        make_shape(Int<4>{}, Int<6>{}),
        GenColMajor{}
    );
    
    std::cout << "Column-Major Layout: ";
    print(col_layout);
    std::cout << std::endl;
    
    int offset_ex3 = col_layout(2, 3);
    std::cout << "Offset at position (2, 3): " << offset_ex3 << std::endl;
    
    std::cout << "Verification:" << std::endl;
    std::cout << "  Expected: 2*1 + 3*4 = 14" << std::endl;
    std::cout << "  Got:      " << offset_ex3 << std::endl;
    std::cout << "  Match: " << (offset_ex3 == 14 ? "YES" : "NO") << std::endl;
    
    // Show column-major access pattern
    std::cout << "\nColumn-major access pattern (first 3 columns):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("Row %d: ", i);
        for (int j = 0; j < 3 && j < 6; ++j) {
            printf("%2d  ", col_layout(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Exercise 1: Created 6x8 row-major layout with stride (8, 1)" << std::endl;
    std::cout << "Exercise 2: Created padded 5x5 layout, offset(3,4) = 28" << std::endl;
    std::cout << "Exercise 3: Created 4x6 column-major layout, offset(2,3) = 14" << std::endl;
    std::cout << std::endl;
    std::cout << "Key Takeaways:" << std::endl;
    std::cout << "  - Row-major: stride = [N, 1] for MxN matrix" << std::endl;
    std::cout << "  - Column-major: stride = [1, M] for MxN matrix" << std::endl;
    std::cout << "  - Padding: customize stride to add spacing between rows" << std::endl;
    std::cout << "  - offset(i,j) = i*stride[0] + j*stride[1]" << std::endl;

    return 0;
}
