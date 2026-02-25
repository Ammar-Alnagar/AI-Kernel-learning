/**
 * @file exercise_02.cu
 * @brief Exercise: Tensor Creation and Views
 * 
 * Complete the following exercises:
 * 
 * Exercise 1: Create a 5x5 tensor with values 1-25 in row-major order.
 *             Print the tensor and verify the shape.
 * 
 * Exercise 2: From the tensor in Exercise 1, create a view of row 3.
 *             Print the row values.
 * 
 * Exercise 3: Create a column-major view of the same data.
 *             Access element at position (2, 3) in both layouts and compare.
 * 
 * Exercise 4: Compute the sum of all elements in the tensor.
 * 
 * Instructions:
 * 1. Fill in the TODO sections
 * 2. Build and run the solution to verify
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== Exercise 02: Tensor Practice ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 1: Create a 5x5 tensor
    // ================================================================
    std::cout << "--- Exercise 1: Create 5x5 Tensor ---" << std::endl;
    
    // TODO: Create data vector with 25 elements (values 1-25)
    // std::vector<float> data(25);
    // for (int i = 0; i < 25; ++i) { data[i] = ...; }
    
    // TODO: Create row-major layout for 5x5
    // auto layout = make_layout(...);
    
    // TODO: Create tensor
    // auto tensor = make_tensor(...);
    
    std::cout << "TODO: Create and print 5x5 tensor" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 2: Create a row view
    // ================================================================
    std::cout << "--- Exercise 2: Row View ---" << std::endl;
    
    // TODO: Extract row 3 from the tensor
    // auto row_3 = tensor(3, _);
    
    std::cout << "TODO: Create and print row 3 view" << std::endl;
    std::cout << "Expected: 16, 17, 18, 19, 20" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 3: Column-major view comparison
    // ================================================================
    std::cout << "--- Exercise 3: Column-Major Comparison ---" << std::endl;
    
    // TODO: Create column-major layout for 5x5
    // auto col_layout = make_layout(...);
    
    // TODO: Create column-major tensor view of same data
    // auto col_tensor = make_tensor(...);
    
    // TODO: Access element at (2, 3) in both layouts
    // float row_major_val = tensor(2, 3);
    // float col_major_val = col_tensor(2, 3);
    
    std::cout << "TODO: Compare row-major and column-major access at (2, 3)" << std::endl;
    std::cout << "Expected row-major offset: 2*5 + 3 = 13, value = 14" << std::endl;
    std::cout << "Expected col-major offset: 2 + 3*5 = 17, value = 18" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 4: Sum reduction
    // ================================================================
    std::cout << "--- Exercise 4: Sum Reduction ---" << std::endl;
    
    // TODO: Compute sum of all elements
    // float sum = 0;
    // for (int i = 0; i < 5; ++i) {
    //     for (int j = 0; j < 5; ++j) {
    //         sum += tensor(i, j);
    //     }
    // }
    
    std::cout << "TODO: Compute sum of all elements" << std::endl;
    std::cout << "Expected: 1+2+...+25 = " << (25 * 26 / 2) << std::endl;
    std::cout << std::endl;

    // ================================================================
    std::cout << "=== Exercise Complete ===" << std::endl;
    std::cout << "See exercise_02_solution.cu for the complete solution" << std::endl;

    return 0;
}
