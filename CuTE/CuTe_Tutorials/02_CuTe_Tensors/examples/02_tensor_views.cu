/**
 * @file 02_tensor_views.cu
 * @brief Tensor Views - Slicing, Dicing, and Reshaping
 * 
 * This tutorial demonstrates tensor view operations:
 * - Creating views without copying data
 * - Slicing and dicing tensors
 * - Reshaping tensors
 * - Transposing views
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>

using namespace cute;

int main() {
    std::cout << "=== CuTe Tutorial: Tensor Views ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: What is a Tensor View?
    // ================================================================
    std::cout << "--- Concept 1: Tensor Views ---" << std::endl;
    std::cout << "A tensor view is a different interpretation of the same data." << std::endl;
    std::cout << "Views do NOT copy data - they just change how we access it." << std::endl;
    std::cout << "This is efficient and enables zero-copy transformations." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Setup: Create a base tensor
    // ================================================================
    std::vector<float> data(24);
    for (int i = 0; i < 24; ++i) {
        data[i] = static_cast<float>(i + 1);
    }
    
    auto base_layout = make_layout(make_shape(Int<4>{}, Int<6>{}), GenRowMajor{});
    auto base_tensor = make_tensor(data.data(), base_layout);
    
    std::cout << "Base tensor (4x6):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 6; ++j) {
            printf("%5.1f ", base_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 1: Row Slice
    // ================================================================
    std::cout << "--- Example 1: Row Slice ---" << std::endl;
    std::cout << "Extract row 2 from the base tensor." << std::endl;
    std::cout << std::endl;
    
    // Fix the first dimension to 2, let the second vary
    auto row_2 = base_tensor(2, _);  // _ means "all" in that dimension
    
    std::cout << "Row 2 view:" << std::endl;
    std::cout << "  Shape: ";
    print(row_2.shape());
    std::cout << std::endl;
    std::cout << "  Values: ";
    for (int j = 0; j < 6; ++j) {
        printf("%5.1f ", row_2(j));
    }
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Column Slice
    // ================================================================
    std::cout << "--- Example 2: Column Slice ---" << std::endl;
    std::cout << "Extract column 3 from the base tensor." << std::endl;
    std::cout << std::endl;
    
    // Fix the second dimension to 3, let the first vary
    auto col_3 = base_tensor(_, 3);
    
    std::cout << "Column 3 view:" << std::endl;
    std::cout << "  Shape: ";
    print(col_3.shape());
    std::cout << std::endl;
    std::cout << "  Values: ";
    for (int i = 0; i < 4; ++i) {
        printf("%5.1f\n", col_3(i));
    }
    std::cout << std::endl;

    // ================================================================
    // Example 3: Sub-tensor (Region of Interest)
    // ================================================================
    std::cout << "--- Example 3: Sub-tensor (ROI) ---" << std::endl;
    std::cout << "Extract a 2x3 sub-region starting at (1, 2)." << std::endl;
    std::cout << std::endl;
    
    // In CuTe, we can create a view with offset
    // Using make_tensor with adjusted pointer and new layout
    auto sub_layout = make_layout(make_shape(Int<2>{}, Int<3>{}), GenRowMajor{});
    auto sub_tensor = make_tensor(base_tensor.data() + base_layout(1, 2), sub_layout);
    
    std::cout << "Sub-tensor (2x3 starting at 1,2):" << std::endl;
    for (int i = 0; i < 2; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", sub_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Original tensor at same region:" << std::endl;
    for (int i = 0; i < 2; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", base_tensor(i + 1, j + 2));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 4: Reshape View
    // ================================================================
    std::cout << "--- Example 4: Reshape View ---" << std::endl;
    std::cout << "Reshape the 4x6 tensor to 8x3 (same 24 elements)." << std::endl;
    std::cout << std::endl;
    
    auto reshaped_layout = make_layout(make_shape(Int<8>{}, Int<3>{}), GenRowMajor{});
    auto reshaped_tensor = make_tensor(data.data(), reshaped_layout);
    
    std::cout << "Reshaped tensor (8x3):" << std::endl;
    for (int i = 0; i < 8; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", reshaped_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "Same data, different logical organization!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Transpose View
    // ================================================================
    std::cout << "--- Example 5: Transpose View ---" << std::endl;
    std::cout << "Create a transposed view of the base tensor." << std::endl;
    std::cout << std::endl;
    
    // Create transposed layout (swap dimensions)
    auto transposed_layout = make_layout(make_shape(Int<6>{}, Int<4>{}), GenRowMajor{});
    auto transposed_tensor = make_tensor(data.data(), transposed_layout);
    
    std::cout << "Transposed tensor (6x4):" << std::endl;
    for (int i = 0; i < 6; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 4; ++j) {
            printf("%5.1f ", transposed_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 6: Strided View
    // ================================================================
    std::cout << "--- Example 6: Strided View ---" << std::endl;
    std::cout << "Create a view with strided access (every other element)." << std::endl;
    std::cout << std::endl;
    
    // Create a layout with stride 2 in the column dimension
    auto strided_layout = make_layout(
        make_shape(Int<4>{}, Int<3>{}),  // 4 rows, 3 "logical" columns
        make_stride(Int<6>{}, Int<2>{})   // stride of 6 for rows, 2 for columns
    );
    auto strided_tensor = make_tensor(data.data(), strided_layout);
    
    std::cout << "Strided tensor (every other column):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        printf("  Row %d: ", i);
        for (int j = 0; j < 3; ++j) {
            printf("%5.1f ", strided_tensor(i, j));
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;
    
    std::cout << "This accesses columns 0, 2, 4 from the original data." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 7: View Properties
    // ================================================================
    std::cout << "--- Example 7: View Properties ---" << std::endl;
    
    std::cout << "Comparing base tensor and views:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Base tensor:" << std::endl;
    std::cout << "  Data pointer: " << static_cast<void*>(base_tensor.data()) << std::endl;
    std::cout << "  Shape: ";
    print(base_tensor.shape());
    std::cout << std::endl;
    std::cout << std::endl;
    
    std::cout << "Row 2 view:" << std::endl;
    std::cout << "  Data pointer: " << static_cast<void*>(row_2.data()) << std::endl;
    std::cout << "  Shape: ";
    print(row_2.shape());
    std::cout << std::endl;
    std::cout << "  Note: Same data pointer, different shape!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Views provide different interpretations of the same data" << std::endl;
    std::cout << "2. Use tensor(_, j) to slice rows, tensor(i, _) to slice columns" << std::endl;
    std::cout << "3. Reshape changes logical dimensions without copying" << std::endl;
    std::cout << "4. Transpose swaps dimensions for different access patterns" << std::endl;
    std::cout << "5. Strided views skip elements in memory" << std::endl;
    std::cout << "6. Views are zero-copy - efficient and fast!" << std::endl;
    std::cout << std::endl;

    return 0;
}
