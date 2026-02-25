/**
 * @file exercise_01.cu
 * @brief Exercise: Practice Layout Algebra
 * 
 * Complete the following exercises to practice layout concepts:
 * 
 * Exercise 1: Create a 6x8 row-major layout and print its shape and stride.
 * 
 * Exercise 2: Create a padded layout for a 5x5 matrix with stride 8 (3 padding elements per row).
 *             Print the memory offset for position (3, 4).
 * 
 * Exercise 3: Create a column-major layout for a 4x6 matrix.
 *             Verify that layout(2, 3) returns the correct offset.
 * 
 * Instructions:
 * 1. Fill in the TODO sections below
 * 2. Build and run: ./tutorial_build/tutorials/exercise_01_solution
 * 3. Compare your output with the expected values
 */

#include <cute/layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== Exercise 01: Layout Algebra Practice ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 1: Create a 6x8 row-major layout
    // ================================================================
    std::cout << "--- Exercise 1: 6x8 Row-Major Layout ---" << std::endl;
    
    // TODO: Create a row-major layout with shape (6, 8)
    // Hint: Use make_layout(make_shape(Int<6>{}, Int<8>{}), GenRowMajor{})
    // auto layout_ex1 = ...;
    
    std::cout << "TODO: Create and print the layout" << std::endl;
    std::cout << "Expected shape: (6, 8)" << std::endl;
    std::cout << "Expected stride: (8, 1)" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 2: Create a padded 5x5 layout with row stride 8
    // ================================================================
    std::cout << "--- Exercise 2: Padded 5x5 Layout ---" << std::endl;
    
    // TODO: Create a layout with shape (5, 5) and stride (8, 1)
    // Hint: Use make_layout(make_shape(Int<5>{}, Int<5>{}), make_stride(Int<8>{}, Int<1>{}))
    // auto padded_layout = ...;
    
    // TODO: Calculate the offset for position (3, 4)
    // int offset = padded_layout(3, 4);
    
    std::cout << "TODO: Create padded layout and calculate offset at (3, 4)" << std::endl;
    std::cout << "Expected offset: 3*8 + 4*1 = 28" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 3: Create a 4x6 column-major layout
    // ================================================================
    std::cout << "--- Exercise 3: 4x6 Column-Major Layout ---" << std::endl;
    
    // TODO: Create a column-major layout with shape (4, 6)
    // Hint: Use make_layout(make_shape(Int<4>{}, Int<6>{}), GenColMajor{})
    // auto col_layout = ...;
    
    // TODO: Calculate the offset for position (2, 3)
    // int offset = col_layout(2, 3);
    
    std::cout << "TODO: Create column-major layout and calculate offset at (2, 3)" << std::endl;
    std::cout << "Expected offset: 2*1 + 3*4 = 14" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Verification (Uncomment after completing exercises)
    // ================================================================
    std::cout << "--- Verification ---" << std::endl;
    std::cout << "After completing the exercises, uncomment the code above" << std::endl;
    std::cout << "and verify your answers match the expected values." << std::endl;
    std::cout << std::endl;
    
    /*
    // Verification code for Exercise 1
    std::cout << "Exercise 1 Results:" << std::endl;
    std::cout << "  Layout: ";
    print(layout_ex1);
    std::cout << std::endl;
    
    // Verification code for Exercise 2
    std::cout << "Exercise 2 Results:" << std::endl;
    std::cout << "  Padded layout offset at (3, 4): " << offset << std::endl;
    
    // Verification code for Exercise 3
    std::cout << "Exercise 3 Results:" << std::endl;
    std::cout << "  Column-major offset at (2, 3): " << offset << std::endl;
    */

    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "Compare your solutions with exercise_01_solution.cu" << std::endl;

    return 0;
}
