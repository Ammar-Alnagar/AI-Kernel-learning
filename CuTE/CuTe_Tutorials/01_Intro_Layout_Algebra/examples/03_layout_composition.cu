/**
 * @file 03_layout_composition.cu
 * @brief Layout Composition and Transformation
 * 
 * This tutorial covers advanced layout operations:
 * - Layout composition (combining layouts)
 * - Layout transformation (transpose, reshape)
 * - Hierarchical layouts
 * - Z-ordering and swizzling basics
 */

#include <cute/layout.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
    std::cout << "=== CuTe Tutorial: Layout Composition ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: Layout Composition
    // ================================================================
    std::cout << "--- Concept 1: Layout Composition ---" << std::endl;
    std::cout << "Layouts can be composed to create hierarchical structures." << std::endl;
    std::cout << "This is fundamental for tiling and thread block organization." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Simple Layout Composition
    // ================================================================
    std::cout << "--- Example 1: Composing Two Layouts ---" << std::endl;
    
    // Create a base layout (4x4 matrix)
    auto base_layout = make_layout(
        make_shape(Int<4>{}, Int<4>{}),
        GenRowMajor{}
    );
    
    std::cout << "Base layout (4x4): ";
    print(base_layout);
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Tiled Layout (Outer x Inner)
    // ================================================================
    std::cout << "--- Example 2: Tiled Layout ---" << std::endl;
    std::cout << "Divide a 8x8 matrix into 2x2 tiles, each tile is 4x4 elements." << std::endl;
    std::cout << std::endl;
    
    // Outer layout: 2x2 tiles
    auto outer_layout = make_layout(
        make_shape(Int<2>{}, Int<2>{}),
        GenRowMajor{}
    );
    
    // Inner layout: 4x4 elements per tile
    auto inner_layout = make_layout(
        make_shape(Int<4>{}, Int<4>{}),
        GenRowMajor{}
    );
    
    std::cout << "Outer layout (2x2 tiles): ";
    print(outer_layout);
    std::cout << std::endl;
    
    std::cout << "Inner layout (4x4 elements): ";
    print(inner_layout);
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Compose layouts using the comma operator
    // This creates a hierarchical layout
    auto tiled_layout = composition(outer_layout, inner_layout);
    
    std::cout << "Composed tiled layout: ";
    print(tiled_layout);
    std::cout << std::endl;
    std::cout << "Total elements: " << size(tiled_layout) << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Transpose Layout
    // ================================================================
    std::cout << "--- Example 3: Transpose Layout ---" << std::endl;
    
    auto original = make_layout(
        make_shape(Int<3>{}, Int<4>{}),
        GenRowMajor{}
    );
    
    std::cout << "Original layout (3x4 row-major): ";
    print(original);
    std::cout << std::endl;
    
    // Transpose by swapping shape and stride
    auto transposed = make_layout(
        make_shape(Int<4>{}, Int<3>{}),  // Swapped shape
        GenRowMajor{}                     // Still row-major, but different shape
    );
    
    std::cout << "Transposed layout (4x3 row-major): ";
    print(transposed);
    std::cout << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Reshape Layout
    // ================================================================
    std::cout << "--- Example 4: Reshape Layout ---" << std::endl;
    std::cout << "Reshape changes the logical view without changing memory layout." << std::endl;
    std::cout << std::endl;
    
    // Create a 12-element 1D layout
    auto layout_1d = make_layout(make_shape(Int<12>{}), GenRowMajor{});
    std::cout << "1D Layout (12 elements): ";
    print(layout_1d);
    std::cout << std::endl;
    
    // Reshape to 3x4 2D layout (same 12 elements)
    auto layout_2d = make_layout(make_shape(Int<3>{}, Int<4>{}), GenRowMajor{});
    std::cout << "2D Layout (3x4 = 12 elements): ";
    print(layout_2d);
    std::cout << std::endl;
    
    // Reshape to 2x2x3 3D layout (same 12 elements)
    auto layout_3d = make_layout(make_shape(Int<2>{}, Int<2>{}, Int<3>{}), GenRowMajor{});
    std::cout << "3D Layout (2x2x3 = 12 elements): ";
    print(layout_3d);
    std::cout << std::endl;
    std::cout << std::endl;
    
    std::cout << "All three layouts have the same size but different logical views." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Hierarchical Layout for Thread Mapping
    // ================================================================
    std::cout << "--- Example 5: Hierarchical Layout (Thread Mapping) ---" << std::endl;
    std::cout << "Hierarchical layouts are used to map threads to data." << std::endl;
    std::cout << std::endl;
    
    // Thread layout: 4 threads
    auto thread_layout = make_layout(make_shape(Int<4>{}));
    
    // Data layout per thread: 4 elements
    auto data_per_thread = make_layout(make_shape(Int<4>{}));
    
    std::cout << "Thread layout (4 threads): ";
    print(thread_layout);
    std::cout << std::endl;
    
    std::cout << "Data per thread (4 elements): ";
    print(data_per_thread);
    std::cout << std::endl;
    std::cout << std::endl;
    
    // Compose to get thread-data mapping
    auto thread_data_layout = composition(thread_layout, data_per_thread);
    
    std::cout << "Thread-Data composed layout: ";
    print(thread_data_layout);
    std::cout << std::endl;
    std::cout << "This maps 4 threads x 4 elements = 16 total data elements" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: Visualizing Hierarchical Layout
    // ================================================================
    std::cout << "--- Example 6: Visualizing Hierarchy ---" << std::endl;

    // Create a simple 2D layout to demonstrate tiling concept
    auto tiling_layout = make_layout(make_shape(Int<6>{}, Int<4>{}), GenRowMajor{});

    std::cout << "Layout (6x4): ";
    print(tiling_layout);
    std::cout << std::endl;

    std::cout << "Tiling concept: Divide into 2x2 tiles" << std::endl;
    std::cout << "  Each tile is 3x2 elements" << std::endl;
    std::cout << "  Total: 2x2 = 4 tiles" << std::endl;
    std::cout << std::endl;

    std::cout << "Memory layout with tile boundaries:" << std::endl;
    for (int i = 0; i < 6; ++i) {
        for (int j = 0; j < 4; ++j) {
            int offset = tiling_layout(i, j);
            // Add tile separator
            if (j == 2) std::cout << " | ";
            printf("%2d ", offset);
        }
        // Add horizontal separator
        if (i == 2) {
            std::cout << std::endl;
            for (int k = 0; k < 4; ++k) {
                if (k == 2) std::cout << "-+-";
                else std::cout << "---";
            }
        }
        std::cout << std::endl;
    }
    std::cout << std::endl;

    // ================================================================
    // Example 7: Layout Arithmetic
    // ================================================================
    std::cout << "--- Example 7: Layout Operations ---" << std::endl;
    
    auto layout_a = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
    auto layout_b = make_layout(make_shape(Int<4>{}, Int<4>{}), GenColMajor{});
    
    std::cout << "Layout A (row-major): ";
    print(layout_a);
    std::cout << std::endl;
    
    std::cout << "Layout B (col-major): ";
    print(layout_b);
    std::cout << std::endl;
    std::cout << std::endl;
    
    std::cout << "Both layouts have the same shape but different strides." << std::endl;
    std::cout << "This affects memory access patterns and performance." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Layout composition creates hierarchical structures" << std::endl;
    std::cout << "2. Tiled layouts = outer_tiles x inner_elements" << std::endl;
    std::cout << "3. Transpose changes shape interpretation" << std::endl;
    std::cout << "4. Reshape changes logical view, not memory layout" << std::endl;
    std::cout << "5. Hierarchical layouts map threads to data" << std::endl;
    std::cout << "6. Use composition() to combine layouts" << std::endl;
    std::cout << "7. Same shape + different stride = different access patterns" << std::endl;
    std::cout << std::endl;

    return 0;
}
