#include <iostream>
#include <vector>
#include "cutlass/cutlass.h"
#include "cutlass/array.h"
#include "cute/layout.hpp"
#include "cute/shape.hpp"
#include "cute/tensor.hpp"
#include "cute/print.hpp"

using namespace cute;

// Function to demonstrate how a layout maps logical coordinates to memory offsets
void demonstrate_layout_mapping() {
    std::cout << "\n=== Layout Mapping Demonstration ===" << std::endl;
    
    // Create a simple 4x3 row-major layout (like C matrix)
    // Shape: 4 rows x 3 columns
    // Stride: [3, 1] - stride of 3 for rows, 1 for columns
    auto layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenRowMajor{});
    
    std::cout << "Layout: " << layout << std::endl;
    std::cout << "Shape: " << layout.shape() << std::endl;
    std::cout << "Stride: " << layout.stride() << std::endl;
    
    std::cout << "\nMapping logical coordinates to memory offsets:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            int offset = layout(i, j);
            std::cout << "layout(" << i << ", " << j << ") = " << offset << std::endl;
        }
    }
}

// Function to demonstrate column-major layout
void demonstrate_column_major() {
    std::cout << "\n=== Column-Major Layout ===" << std::endl;
    
    // Column-major layout: stride of 1 for rows, 4 for columns
    auto col_major_layout = make_layout(make_shape(Int<4>{}, Int<3>{}), GenColMajor{});
    
    std::cout << "Column-Major Layout: " << col_major_layout << std::endl;
    std::cout << "Shape: " << col_major_layout.shape() << std::endl;
    std::cout << "Stride: " << col_major_layout.stride() << std::endl;
    
    std::cout << "\nMapping logical coordinates to memory offsets:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            int offset = col_major_layout(i, j);
            std::cout << "layout(" << i << ", " << j << ") = " << offset << std::endl;
        }
    }
}

// Function to demonstrate custom strides
void demonstrate_custom_strides() {
    std::cout << "\n=== Custom Stride Layout (Padded Matrix) ===" << std::endl;
    
    // Layout with padding: 4x3 matrix but with stride 5 for rows (padding of 2 elements)
    // This simulates a padded matrix where each row has extra space
    auto padded_layout = make_layout(make_shape(Int<4>{}, Int<3>{}), make_stride(Int<5>{}, Int<1>{}));
    
    std::cout << "Padded Layout: " << padded_layout << std::endl;
    std::cout << "Shape: " << padded_layout.shape() << std::endl;
    std::cout << "Stride: " << padded_layout.stride() << std::endl;
    
    std::cout << "\nMapping logical coordinates to memory offsets (with padding):" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 3; ++j) {
            int offset = padded_layout(i, j);
            std::cout << "layout(" << i << ", " << j << ") = " << offset << std::endl;
        }
    }
}

// Function to demonstrate 2D layout visualization
void visualize_2d_layout() {
    std::cout << "\n=== 2D Layout Visualization ===" << std::endl;
    
    // Create a 4x6 layout to visualize
    auto layout = make_layout(make_shape(Int<4>{}, Int<6>{}), GenRowMajor{});
    
    std::cout << "Visualizing 4x6 Row-Major Layout:" << std::endl;
    std::cout << "Layout: " << layout << std::endl;
    
    // Print the memory layout in a grid format
    std::cout << "\nMemory layout grid:" << std::endl;
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 6; ++j) {
            int offset = layout(i, j);
            printf("%3d ", offset);
        }
        std::cout << std::endl;
    }
    
    // Show how cute::print can be used for debugging
    std::cout << "\nUsing cute::print for debugging:" << std::endl;
    print(layout);  // This is the key debugging function
    std::cout << std::endl;
}

// Function to demonstrate hierarchical layouts
void demonstrate_hierarchical_layout() {
    std::cout << "\n=== Hierarchical Layout Example ===" << std::endl;
    
    // Create a 2D layout and tile it into 2x3 tiles
    auto base_layout = make_layout(make_shape(Int<4>{}, Int<6>{}), GenRowMajor{});
    auto tile_layout = make_layout(make_shape(Int<2>{}, Int<3>{}), GenRowMajor{});
    
    std::cout << "Base layout (4x6): " << base_layout << std::endl;
    std::cout << "Tile shape (2x3): " << tile_layout << std::endl;
    
    // Apply tiling to create a hierarchical layout
    auto tiled_layout = zipped_divide(base_layout, make_shape(tile_layout.shape()));
    
    std::cout << "Tiled layout result: " << tiled_layout << std::endl;
    
    // Show how cute::print helps debug complex layouts
    std::cout << "\nDebug view of base layout:" << std::endl;
    print(base_layout);
    std::cout << std::endl;
}

int main() {
    std::cout << "CuTe Layout Algebra Study - Module 01" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    demonstrate_layout_mapping();
    demonstrate_column_major();
    demonstrate_custom_strides();
    visualize_2d_layout();
    demonstrate_hierarchical_layout();
    
    std::cout << "\n=== Summary ===" << std::endl;
    std::cout << "1. Layouts map logical coordinates (i,j) to memory offsets" << std::endl;
    std::cout << "2. Shape defines dimensions, Stride defines step sizes" << std::endl;
    std::cout << "3. cute::print() is essential for debugging layout mappings" << std::endl;
    std::cout << "4. Hierarchical layouts enable complex tiling strategies" << std::endl;
    
    return 0;
}