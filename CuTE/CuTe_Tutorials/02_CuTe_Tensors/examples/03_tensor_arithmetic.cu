/**
 * @file 03_tensor_arithmetic.cu
 * @brief Tensor Arithmetic Operations
 * 
 * This tutorial demonstrates basic tensor arithmetic:
 * - Element-wise operations
 * - Tensor addition and multiplication
 * - Scaling tensors
 * - Reduction operations
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>
#include <cmath>

using namespace cute;

int main() {
    std::cout << "=== CuTe Tutorial: Tensor Arithmetic ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Setup: Create test tensors
    // ================================================================
    std::vector<float> data_a(12);
    std::vector<float> data_b(12);
    
    for (int i = 0; i < 12; ++i) {
        data_a[i] = static_cast<float>(i + 1);
        data_b[i] = static_cast<float>((i + 1) * 2);
    }
    
    auto layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenRowMajor{});
    auto tensor_a = make_tensor(data_a.data(), layout);
    auto tensor_b = make_tensor(data_b.data(), layout);
    
    std::cout << "Tensor A (4x3):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", tensor_a(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Tensor B (4x3):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", tensor_b(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 1: Element-wise Addition
    // ================================================================
    std::cout << "--- Example 1: Element-wise Addition ---" << std::endl;
    
    std::vector<float> data_result(12);
    auto tensor_result = make_tensor(data_result.data(), layout);
    
    // Manual element-wise addition
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor_result(i, j) = tensor_a(i, j) + tensor_b(i, j);
        }
    }
    
    std::cout << "A + B:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", tensor_result(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 2: Element-wise Multiplication
    // ================================================================
    std::cout << "--- Example 2: Element-wise Multiplication ---" << std::endl;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor_result(i, j) = tensor_a(i, j) * tensor_b(i, j);
        }
    }
    
    std::cout << "A * B (element-wise):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%7.1f ", tensor_result(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 3: Scalar Multiplication
    // ================================================================
    std::cout << "--- Example 3: Scalar Multiplication ---" << std::endl;
    
    float scalar = 2.5f;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor_result(i, j) = tensor_a(i, j) * scalar;
        }
    }
    
    std::cout << "A * " << scalar << ":" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%6.1f ", tensor_result(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 4: Element-wise Subtraction
    // ================================================================
    std::cout << "--- Example 4: Element-wise Subtraction ---" << std::endl;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor_result(i, j) = tensor_b(i, j) - tensor_a(i, j);
        }
    }
    
    std::cout << "B - A:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", tensor_result(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 5: Sum Reduction
    // ================================================================
    std::cout << "--- Example 5: Sum Reduction ---" << std::endl;
    
    float sum = 0.0f;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            sum += tensor_a(i, j);
        }
    }
    
    std::cout << "Sum of all elements in A: " << sum << std::endl;
    std::cout << "Expected (1+2+...+12): " << (12 * 13 / 2) << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: Row-wise Sum Reduction
    // ================================================================
    std::cout << "--- Example 6: Row-wise Sum Reduction ---" << std::endl;
    
    std::vector<float> row_sums(4);
    
    for (int i = 0; i < 4; ++i) {
        row_sums[i] = 0.0f;
        for (int j = 0; j < 3; ++j) {
            row_sums[i] += tensor_a(i, j);
        }
    }
    
    std::cout << "Row sums:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        std::cout << "  Row " << i << " sum: " << row_sums[i] << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 7: Column-wise Sum Reduction
    // ================================================================
    std::cout << "--- Example 7: Column-wise Sum Reduction ---" << std::endl;
    
    std::vector<float> col_sums(3);
    
    for (int j = 0; j < 3; ++j) {
        col_sums[j] = 0.0f;
        for (int i = 0; i < 4; ++i) {
            col_sums[j] += tensor_a(i, j);
        }
    }
    
    std::cout << "Column sums:" << std::endl;
    for (int j = 0; j < 3; ++j) {
        std::cout << "  Column " << j << " sum: " << col_sums[j] << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 8: Element-wise Division
    // ================================================================
    std::cout << "--- Example 8: Element-wise Division ---" << std::endl;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor_result(i, j) = tensor_b(i, j) / tensor_a(i, j);
        }
    }
    
    std::cout << "B / A:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%6.2f ", tensor_result(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 9: Fused Operation (axpy: y = a*x + y)
    // ================================================================
    std::cout << "--- Example 9: Fused Operation (AXPY) ---" << std::endl;
    std::cout << "Compute: result = alpha * A + B" << std::endl;
    
    float alpha = 3.0f;
    
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            tensor_result(i, j) = alpha * tensor_a(i, j) + tensor_b(i, j);
        }
    }
    
    std::cout << "3 * A + B:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%7.1f ", tensor_result(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 10: Element-wise Comparison
    // ================================================================
    std::cout << "--- Example 10: Element-wise Comparison ---" << std::endl;
    
    std::cout << "Elements where A > 6:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            if (tensor_a(i, j) > 6) {
                printf("%5.1f ", tensor_a(i, j));
            } else {
                printf("    - ");
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Element-wise operations: loop over all indices" << std::endl;
    std::cout << "2. Addition/Subtraction: element-wise with matching shapes" << std::endl;
    std::cout << "3. Multiplication: element-wise (Hadamard) or scalar" << std::endl;
    std::cout << "4. Reduction: sum over dimensions to get lower-rank result" << std::endl;
    std::cout << "5. Fused operations (like AXPY) combine multiple ops efficiently" << std::endl;
    std::cout << "6. CuTe tensors use operator() for element access" << std::endl;
    std::cout << std::endl;

    return 0;
}
