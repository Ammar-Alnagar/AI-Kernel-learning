/**
 * @file 01_layout_basics.cu
 * @brief Introduction to CuTe Layout - The Foundation of Memory Mapping
 *
 * This tutorial demonstrates the fundamental concept of Layout in CuTe:
 * A Layout maps logical coordinates (like row, column) to physical memory
 * offsets.
 *
 * Key Concepts:
 * - Layout = Shape + Stride
 * - Logical coordinate -> Physical offset mapping
 * - Row-major vs Column-major layouts
 */

#include <cute/int_tuple.hpp>
#include <cute/layout.hpp>
#include <cute/util/print.hpp>
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== CuTe Tutorial: Layout Basics ===" << std::endl;
  std::cout << std::endl;

  // ================================================================
  // Concept 1: What is a Layout?
  // ================================================================
  std::cout << "--- Concept 1: What is a Layout? ---" << std::endl;
  std::cout << "A Layout maps logical coordinates to memory offsets."
            << std::endl;
  std::cout << "Layout = Shape (dimensions) + Stride (step sizes)" << std::endl;
  std::cout << std::endl;

  // ================================================================
  // Example 1: Creating a Simple Row-Major Layout
  // ================================================================
  std::cout << "--- Example 1: 4x3 Row-Major Layout ---" << std::endl;

  // Create a 4x3 row-major layout
  // Shape: 4 rows, 3 columns
  // Stride: [3, 1] - move 3 elements to go to next row, 1 for next column
  auto row_major_layout =
      make_layout(make_shape(Int<4>{}, Int<3>{}), // Shape: 4 rows x 3 cols
                  GenRowMajor{}                   // Row-major stride generation
      );

  std::cout << "Layout: ";
  print(row_major_layout);
  std::cout << std::endl;

  std::cout << "Shape:  ";
  print(row_major_layout.shape());
  std::cout << std::endl;

  std::cout << "Stride: ";
  print(row_major_layout.stride());
  std::cout << std::endl;
  std::cout << std::endl;

  // ================================================================
  // Example 2: Mapping Coordinates to Offsets
  // ================================================================
  std::cout << "--- Example 2: Coordinate to Offset Mapping ---" << std::endl;
  std::cout << "Calling layout(i, j) returns the memory offset:" << std::endl;

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      int offset = row_major_layout(i, j);
      std::cout << "  layout(" << i << ", " << j << ") = " << offset;
      if (j == 0)
        std::cout << "  (start of row " << i << ")";
      std::cout << std::endl;
    }
  }
  std::cout << std::endl;

  // ================================================================
  // Example 3: Column-Major Layout
  // ================================================================
  std::cout << "--- Example 3: Column-Major Layout ---" << std::endl;

  auto col_major_layout =
      make_layout(make_shape(Int<4>{}, Int<3>{}), // Same shape: 4 rows x 3 cols
                  GenColMajor{}                   // Column-major stride
      );

  std::cout << "Column-Major Layout: ";
  print(col_major_layout);
  std::cout << std::endl;
  std::cout << "Stride: ";
  print(col_major_layout.stride());
  std::cout << std::endl;

  std::cout << "Notice: In column-major, stride[0]=1 (consecutive rows),"
            << std::endl;
  std::cout << "         while stride[1]=4 (jump 4 elements for next column)"
            << std::endl;
  std::cout << std::endl;

  // ================================================================
  // Example 4: Visualizing Memory Order
  // ================================================================
  std::cout << "--- Example 4: Visual Memory Layout (4x3 Row-Major) ---"
            << std::endl;
  std::cout << "Logical grid with memory offsets:" << std::endl;
  std::cout << std::endl;

  // Print column headers
  std::cout << "      ";
  for (int j = 0; j < 3; ++j) {
    printf("Col%-2d ", j);
  }
  std::cout << std::endl;

  std::cout << "      ";
  for (int j = 0; j < 3; ++j)
    std::cout << "----- ";
  std::cout << std::endl;

  // Print the grid
  for (int i = 0; i < 4; ++i) {
    printf("Row %d | ", i);
    for (int j = 0; j < 3; ++j) {
      printf("%3d   ", row_major_layout(i, j));
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;

  std::cout
      << "Memory is laid out sequentially: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11"
      << std::endl;
  std::cout << std::endl;

  // ================================================================
  // Example 5: Layout Size
  // ================================================================
  std::cout << "--- Example 5: Layout Size ---" << std::endl;
  auto layout_size = size(row_major_layout);
  std::cout << "Total elements in layout: " << layout_size << std::endl;
  std::cout << "(4 rows x 3 columns = 12 elements)" << std::endl;
  std::cout << std::endl;

  // ================================================================
  // Summary
  // ================================================================
  std::cout << "=== Summary ===" << std::endl;
  std::cout << "1. Layout = Shape + Stride" << std::endl;
  std::cout << "2. Row-major: stride = [N, 1] for MxN matrix" << std::endl;
  std::cout << "3. Column-major: stride = [1, M] for MxN matrix" << std::endl;
  std::cout << "4. layout(i, j) computes: i*stride[0] + j*stride[1]"
            << std::endl;
  std::cout << "5. Use print(layout) to debug layout structure" << std::endl;
  std::cout << std::endl;

  return 0;
}
