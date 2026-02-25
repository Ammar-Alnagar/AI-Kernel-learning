/**
 * @file exercise_02_solution.cu
 * @brief Solution: Tensor Creation and Views
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== Exercise 02: Solution ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 1: Create a 5x5 tensor
    // ================================================================
    std::cout << "--- Exercise 1: Create 5x5 Tensor ---" << std::endl;
    
    std::vector<float> data(25);
    for (int i = 0; i < 25; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    auto layout = make_layout(make_shape(Int<5>{}, Int<5>{}), GenRowMajor{});
    auto tensor = make_tensor(data.data(), layout);
    
    std::cout << "5x5 Tensor:" << std::endl;
    for (int i = 0; i < 5; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 5; ++j) {
            printf("%5.1f ", tensor(i, j));
        }
        std::cout << std::endl;
    }
    
    std::cout << "Shape: ";
    print(tensor.shape());
    std::cout << std::endl;
    std::cout << "Verification: Shape is (5, 5) - " 
              << (get<0>(tensor.shape()) == 5 && get<1>(tensor.shape()) == 5 ? "CORRECT" : "INCORRECT")
              << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 2: Create a row view
    // ================================================================
    std::cout << "--- Exercise 2: Row View ---" << std::endl;
    
    auto row_3 = tensor(3, _);
    
    std::cout << "Row 3 view:" << std::endl;
    std::cout << "  Shape: ";
    print(row_3.shape());
    std::cout << std::endl;
    std::cout << "  Values: ";
    for (int j = 0; j < 5; ++j) {
        printf("%5.1f ", row_3(j));
    }
    std::cout << std::endl;
    
    std::cout << "Expected: 16, 17, 18, 19, 20" << std::endl;
    std::cout << "Match: " << (row_3(0) == 16 && row_3(4) == 20 ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 3: Column-major view comparison
    // ================================================================
    std::cout << "--- Exercise 3: Column-Major Comparison ---" << std::endl;
    
    auto col_layout = make_layout(make_shape(Int<5>{}, Int<5>{}), GenColMajor{});
    auto col_tensor = make_tensor(data.data(), col_layout);
    
    std::cout << "Row-major layout: ";
    print(layout);
    std::cout << std::endl;
    
    std::cout << "Column-major layout: ";
    print(col_layout);
    std::cout << std::endl;
    
    float row_major_val = tensor(2, 3);
    float col_major_val = col_tensor(2, 3);
    
    std::cout << "Element at (2, 3):" << std::endl;
    std::cout << "  Row-major value:    " << row_major_val << std::endl;
    std::cout << "  Column-major value: " << col_major_val << std::endl;
    std::cout << std::endl;
    
    std::cout << "Explanation:" << std::endl;
    std::cout << "  Row-major offset:    2*5 + 3 = 13 -> value = 14" << std::endl;
    std::cout << "  Column-major offset: 2 + 3*5 = 17 -> value = 18" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Verification:" << std::endl;
    std::cout << "  Row-major correct:    " << (row_major_val == 14 ? "YES" : "NO") << std::endl;
    std::cout << "  Column-major correct: " << (col_major_val == 18 ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Exercise 4: Sum reduction
    // ================================================================
    std::cout << "--- Exercise 4: Sum Reduction ---" << std::endl;
    
    float sum = 0.0f;
    for (int i = 0; i < 5; ++i) {
        for (int j = 0; j < 5; ++j) {
            sum += tensor(i, j);
        }
    }
    
    std::cout << "Sum of all elements: " << sum << std::endl;
    std::cout << "Expected (1+2+...+25): " << (25 * 26 / 2) << std::endl;
    std::cout << "Match: " << (sum == 325.0f ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "Exercise 1: Created 5x5 tensor with values 1-25" << std::endl;
    std::cout << "Exercise 2: Extracted row 3 view: [16, 17, 18, 19, 20]" << std::endl;
    std::cout << "Exercise 3: Compared row-major (14) vs column-major (18) at (2,3)" << std::endl;
    std::cout << "Exercise 4: Computed sum = 325" << std::endl;
    std::cout << std::endl;

    return 0;
}
