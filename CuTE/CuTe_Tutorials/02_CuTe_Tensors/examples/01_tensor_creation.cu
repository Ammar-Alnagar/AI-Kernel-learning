/**
 * @file 01_tensor_creation.cu
 * @brief Introduction to CuTe Tensors - Creation and Basics
 * 
 * This tutorial demonstrates how to create tensors in CuTe:
 * - Tensor = Layout + Data pointer
 * - Different ways to create tensors
 * - Accessing tensor elements
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== CuTe Tutorial: Tensor Creation ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: What is a Tensor?
    // ================================================================
    std::cout << "--- Concept 1: What is a Tensor? ---" << std::endl;
    std::cout << "A Tensor in CuTe is a combination of:" << std::endl;
    std::cout << "  1. Layout: Maps logical coordinates to memory offsets" << std::endl;
    std::cout << "  2. Data: Pointer to the actual data in memory" << std::endl;
    std::cout << "  Tensor = Layout + Data pointer" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Creating a Tensor from Raw Data
    // ================================================================
    std::cout << "--- Example 1: Tensor from Raw Data ---" << std::endl;
    
    // Allocate some data
    std::vector<float> data(12);
    for (int i = 0; i < 12; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    // Create a layout for 4x3 matrix
    auto layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenRowMajor{});
    
    // Create a tensor by combining layout and data pointer
    auto tensor = make_tensor(data.data(), layout);
    
    std::cout << "Tensor layout: ";
    print(layout);
    std::cout << std::endl;
    
    std::cout << "Tensor shape:  ";
    print(tensor.shape());
    std::cout << std::endl;
    
    std::cout << "Tensor data (first 5 elements): ";
    for (int i = 0; i < 5; ++i) {
        std::cout << tensor.data()[i] << " ";
    }
    std::cout << "..." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Accessing Tensor Elements
    // ================================================================
    std::cout << "--- Example 2: Accessing Tensor Elements ---" << std::endl;
    
    std::cout << "Accessing elements using (row, col) coordinates:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            std::cout << "tensor(" << i << ", " << j << ") = " << tensor(i, j) << std::endl;
        }
    }
    std::cout << std::endl;

    // ================================================================
    // Example 3: Creating a Tensor with Static Allocation
    // ================================================================
    std::cout << "--- Example 3: Static Tensor Allocation ---" << std::endl;
    
    // Use cute::array for static allocation
    cute::array<float, 12> static_data;
    for (int i = 0; i < 12; ++i) {
        static_data[i] = static_cast<float>((i + 1) * 10);
    }
    
    auto static_tensor = make_tensor(static_data.data(), layout);
    
    std::cout << "Static tensor values:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%6.1f ", static_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 4: Tensor with Different Layouts
    // ================================================================
    std::cout << "--- Example 4: Same Data, Different Layouts ---" << std::endl;
    
    // Same data, different interpretations
    std::vector<float> shared_data(12);
    for (int i = 0; i < 12; ++i) {
        shared_data[i] = static_cast<float>(i + 1);
    }
    
    // Row-major interpretation
    auto row_layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenRowMajor{});
    auto row_tensor = make_tensor(shared_data.data(), row_layout);
    
    // Column-major interpretation
    auto col_layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenColMajor{});
    auto col_tensor = make_tensor(shared_data.data(), col_layout);
    
    std::cout << "Same data, row-major layout:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", row_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Same data, column-major layout:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", col_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Notice how the same memory is interpreted differently!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: 1D Tensor (Vector)
    // ================================================================
    std::cout << "--- Example 5: 1D Tensor (Vector) ---" << std::endl;
    
    std::vector<float> vec_data(8);
    for (int i = 0; i < 8; ++i) {
        vec_data[i] = static_cast<float>((i + 1) * 2);
    }
    
    auto vec_layout = make_layout(make_shape(Int<8>{}));
    auto vec_tensor = make_tensor(vec_data.data(), vec_layout);
    
    std::cout << "1D Tensor (vector of 8):" << std::endl;
    std::cout << "  Shape: ";
    print(vec_tensor.shape());
    std::cout << std::endl;
    std::cout << "  Values: ";
    for (int i = 0; i < 8; ++i) {
        std::cout << vec_tensor(i) << " ";
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: 3D Tensor
    // ================================================================
    std::cout << "--- Example 6: 3D Tensor ---" << std::endl;
    
    std::vector<float> tensor3d_data(24);  // 2 x 3 x 4
    for (int i = 0; i < 24; ++i) {
        tensor3d_data[i] = static_cast<float>(i + 1);
    }
    
    auto layout_3d = make_layout(make_shape(Int<2>{}, Int<3>{}, Int<4>{}), GenRowMajor{});
    auto tensor_3d = make_tensor(tensor3d_data.data(), layout_3d);
    
    std::cout << "3D Tensor (2 x 3 x 4):" << std::endl;
    std::cout << "  Shape: ";
    print(tensor_3d.shape());
    std::cout << std::endl;
    
    std::cout << "  Accessing element at (1, 2, 3): " << tensor_3d(1, 2, 3) << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 7: Tensor Properties
    // ================================================================
    std::cout << "--- Example 7: Tensor Properties ---" << std::endl;
    
    std::cout << "Tensor properties:" << std::endl;
    std::cout << "  Size (total elements): " << size(tensor) << std::endl;
    std::cout << "  Rank (number of dimensions): " << rank(tensor) << std::endl;
    std::cout << "  Shape: ";
    print(tensor.shape());
    std::cout << std::endl;
    std::cout << "  Layout: ";
    print(tensor.layout());
    std::cout << std::endl;
    std::cout << "  Data pointer: " << static_cast<void*>(tensor.data()) << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Tensor = Layout + Data pointer" << std::endl;
    std::cout << "2. Use make_tensor(data_ptr, layout) to create tensors" << std::endl;
    std::cout << "3. Access elements with tensor(i, j, k, ...)" << std::endl;
    std::cout << "4. Same data can have different tensor views via layouts" << std::endl;
    std::cout << "5. Tensors support 1D, 2D, 3D, and higher dimensions" << std::endl;
    std::cout << "6. Properties: size(), rank(), shape(), layout(), data()" << std::endl;
    std::cout << std::endl;

    return 0;
}
