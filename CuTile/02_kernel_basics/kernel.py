# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 02: Kernel Basics - @kernel Decorator, Launch, and Block IDs

INSTRUCTIONS:
1. Read the README.md for concepts about kernels, launch, and block IDs
2. Fill in the code where you see "# FILL IN:" comments
3. Run test.py to verify your solution
4. Compare with solution.py after completing

KEY CONCEPTS:
- @ct.kernel marks GPU functions
- ct.launch(stream, grid, kernel, args) launches kernels
- ct.bid(dim) gets block ID along dimension
- ct.cdiv() for ceiling division
- ct.Constant for compile-time constants
"""

import cuda.tile as ct
import torch
from math import ceil

# Type alias for compile-time constants
ConstInt = ct.Constant[int]

# Tile size for exercises
TILE_SIZE = 32


# =============================================================================
# EXERCISE 1: Complete a Basic 1D Kernel
# =============================================================================
@ct.kernel
def square_kernel(input_array, output_array):
    """
    Square each element of the input array.
    
    Each block processes one tile of the array.
    
    Args:
        input_array: Input array (1D)
        output_array: Output array (1D, same shape)
    """
    # FILL IN: Get the block ID along dimension 0
    block_id =  # FILL IN: ct.bid(?)
    
    # FILL IN: Load a tile from input_array with shape (TILE_SIZE,)
    input_tile =  # FILL IN: ct.load(?, index=(?,), shape=(?,))
    
    # FILL IN: Square the tile (multiply by itself)
    output_tile =  # FILL IN: input_tile ? input_tile
    
    # FILL IN: Store the result tile to output_array
    # FILL IN: ct.store(?, index=(?,), tile=?)
    pass


# =============================================================================
# EXERCISE 2: 2D Grid Kernel for Matrix Operations
# =============================================================================
@ct.kernel
def matrix_add_kernel(matrix_a, matrix_b, output_matrix,
                      tile_m: ConstInt, tile_n: ConstInt):
    """
    Add two matrices using a 2D grid.
    
    Each block processes one 2D tile of the matrices.
    
    Args:
        matrix_a: Input matrix A (2D)
        matrix_b: Input matrix B (2D)
        output_matrix: Output matrix (2D)
        tile_m: Tile size for rows (compile-time constant)
        tile_n: Tile size for columns (compile-time constant)
    """
    # FILL IN: Get block ID for row dimension (dimension 0)
    block_id_m =  # FILL IN: ct.bid(?)
    
    # FILL IN: Get block ID for column dimension (dimension 1)
    block_id_n =  # FILL IN: ct.bid(?)
    
    # FILL IN: Load a 2D tile from matrix_a
    # Shape should be (tile_m, tile_n)
    tile_a =  # FILL IN: ct.load(?, index=(?, ?), shape=(?, ?))
    
    # FILL IN: Load a 2D tile from matrix_b
    tile_b =  # FILL IN: ct.load(?, index=(?, ?), shape=(?, ?))
    
    # FILL IN: Add the tiles
    result_tile =  # FILL IN: tile_a ? tile_b
    
    # FILL IN: Store the result to output_matrix
    # FILL IN: ct.store(?, index=(?, ?), tile=?)
    pass


# =============================================================================
# EXERCISE 3: Calculate Grid Size for Kernel Launch
# =============================================================================
def calculate_grid_1d(array_size: int, tile_size: int) -> tuple:
    """
    Calculate 1D grid size for processing an array.
    
    Args:
        array_size: Total number of elements in the array
        tile_size: Number of elements per tile (power of 2)
    
    Returns:
        Grid tuple (x, y, z) for ct.launch()
    """
    # FILL IN: Calculate number of blocks needed using ceiling division
    # Formula: num_blocks = ceil(array_size / tile_size)
    # Or: num_blocks = (array_size + tile_size - 1) // tile_size
    num_blocks =  # FILL IN: Calculate
    
    # FILL IN: Return grid as 3-tuple (x, y, z)
    # For 1D processing, y and z should be 1
    grid =  # FILL IN: (num_blocks, ?, ?)
    
    return grid


# =============================================================================
# EXERCISE 4: Calculate 2D Grid Size for Matrix Processing
# =============================================================================
def calculate_grid_2d(rows: int, cols: int, 
                      tile_rows: int, tile_cols: int) -> tuple:
    """
    Calculate 2D grid size for processing a matrix.
    
    Args:
        rows: Number of rows in the matrix
        cols: Number of columns in the matrix
        tile_rows: Number of rows per tile
        tile_cols: Number of columns per tile
    
    Returns:
        Grid tuple (x, y, z) for ct.launch()
    """
    # FILL IN: Calculate number of blocks for rows (dimension 0)
    num_blocks_m =  # FILL IN: Ceiling division
    
    # FILL IN: Calculate number of blocks for columns (dimension 1)
    num_blocks_n =  # FILL IN: Ceiling division
    
    # FILL IN: Return grid as 3-tuple
    grid =  # FILL IN: (num_blocks_m, num_blocks_n, ?)
    
    return grid


# =============================================================================
# EXERCISE 5: Host Function to Launch a Kernel
# =============================================================================
def scale_and_add(array_a: torch.Tensor, array_b: torch.Tensor, 
                  scale: float) -> torch.Tensor:
    """
    Compute: result = array_a * scale + array_b
    
    This host function:
    1. Creates output tensor
    2. Calculates grid size
    3. Launches the kernel
    4. Returns result
    
    Args:
        array_a: First input array (1D, CUDA)
        array_b: Second input array (1D, CUDA)
        scale: Scale factor for array_a
    
    Returns:
        result: Output array containing array_a * scale + array_b
    """
    # Validate inputs
    if array_a.shape != array_b.shape:
        raise ValueError("Input arrays must have the same shape")
    if not array_a.is_cuda or not array_b.is_cuda:
        raise ValueError("Input arrays must be on CUDA device")
    
    # FILL IN: Create output tensor with same shape and dtype as inputs
    # Hint: Use torch.empty_like()
    result =  # FILL IN: Create output tensor
    
    # FILL IN: Calculate grid size using calculate_grid_1d
    grid =  # FILL IN: calculate_grid_1d(?, ?)
    
    # FILL IN: Launch the scale_add_kernel (defined below)
    # Use torch.cuda.current_stream() for the stream
    # Pass (array_a, array_b, result, scale) as arguments
    # FILL IN: ct.launch(?, ?, ?, ?)
    
    return result


@ct.kernel
def scale_add_kernel(array_a, array_b, result, scale: float):
    """
    Kernel for computing result = array_a * scale + array_b.
    
    Args:
        array_a: First input array
        array_b: Second input array
        result: Output array
        scale: Scale factor
    """
    # FILL IN: Get block ID
    block_id =  # FILL IN: ct.bid(?)
    
    # FILL IN: Load tiles from both input arrays
    tile_a =  # FILL IN: Load tile from array_a
    tile_b =  # FILL IN: Load tile from array_b
    
    # FILL IN: Compute result = a * scale + b
    result_tile =  # FILL IN: Compute
    
    # FILL IN: Store result tile
    # FILL IN: ct.store(?, index=(?,), tile=?)
    pass


# =============================================================================
# EXERCISE 6: 3D Grid for Batched Operations
# =============================================================================
@ct.kernel
def batch_scale_kernel(batch_array, output_array,
                       batch_size: ConstInt, tile_size: ConstInt,
                       scale: float):
    """
    Scale elements in a batched array using 3D grid.
    
    The array has shape (batch_size, array_size).
    Grid is organized as (batch_idx, tile_idx, 1).
    
    Args:
        batch_array: Input array with shape (batch_size, array_size)
        output_array: Output array (same shape)
        batch_size: Number of batches (compile-time constant)
        tile_size: Size of each tile (compile-time constant)
        scale: Scale factor
    """
    # FILL IN: Get batch ID from dimension 0
    batch_id =  # FILL IN: ct.bid(?)
    
    # FILL IN: Get tile ID within batch from dimension 1
    tile_id =  # FILL IN: ct.bid(?)
    
    # FILL IN: Load a tile from the batch_array
    # Index should be (batch_id, tile_id)
    # Shape should be (1, tile_size) - 1 for batch dimension
    input_tile =  # FILL IN: ct.load(?, index=(?, ?), shape=(?, ?))
    
    # FILL IN: Reshape tile from (1, tile_size) to (tile_size,)
    # Hint: Use ct.reshape(tile, new_shape)
    input_tile_1d =  # FILL IN: ct.reshape(?, ?)
    
    # FILL IN: Apply scaling
    output_tile_1d =  # FILL IN: Apply scale
    
    # FILL IN: Reshape back to (1, tile_size) for storing
    output_tile =  # FILL IN: ct.reshape(?, ?)
    
    # FILL IN: Store the result
    # FILL IN: ct.store(?, index=(?, ?), tile=?)
    pass


# =============================================================================
# EXERCISE 7: Using ct.num_blocks() for Adaptive Processing
# =============================================================================
@ct.kernel
def adaptive_kernel(input_array, output_array, scale: float):
    """
    Process array using ct.num_blocks() for bounds checking.
    
    This demonstrates how to handle cases where the array size
    is not a perfect multiple of the tile size.
    
    Args:
        input_array: Input array (1D)
        output_array: Output array (1D)
        scale: Scale factor
    """
    # FILL IN: Get the current block ID
    block_id =  # FILL IN: ct.bid(?)
    
    # FILL IN: Get the total number of blocks along axis 0
    # Hint: ct.num_blocks(axis)
    total_blocks =  # FILL IN: ct.num_blocks(?)
    
    # FILL IN: Only process if this block is within valid range
    # Use an if statement to check block_id < total_blocks
    # (This is a simplified example - in practice, padding handles boundaries)
    if True:  # FILL IN: Replace with proper condition
        # FILL IN: Load tile with padding mode for boundary handling
        # Hint: Use padding_mode=ct.PaddingMode.ZERO
        input_tile = ct.load(input_array, index=(block_id,), 
                            shape=(TILE_SIZE,),
                            padding_mode=ct.PaddingMode.ZERO)
        
        # FILL IN: Apply scaling
        output_tile =  # FILL IN: Apply scale
        
        # FILL IN: Store result
        # FILL IN: ct.store(?, index=(?,), tile=?)
        pass


# =============================================================================
# Main: Test Your Implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Module 02: Testing Your Kernel Basics Implementation")
    print("=" * 60)
    
    # Test grid calculation
    print("\n[Test 1] Grid Calculation (1D)")
    print("-" * 40)
    
    test_cases_1d = [
        (128, 32, (4, 1, 1)),
        (100, 32, (4, 1, 1)),  # Not a multiple
        (256, 64, (4, 1, 1)),
    ]
    
    for size, tile, expected in test_cases_1d:
        result = calculate_grid_1d(size, tile)
        match = result == expected
        print(f"  Size={size}, Tile={tile}: Expected {expected}, Got {result} - {'✓' if match else '✗'}")
    
    # Test 2D grid calculation
    print("\n[Test 2] Grid Calculation (2D)")
    print("-" * 40)
    
    test_cases_2d = [
        ((64, 64), (32, 32), (2, 2, 1)),
        ((100, 50), (32, 32), (4, 2, 1)),
    ]
    
    for (rows, cols), (tile_rows, tile_cols), expected in test_cases_2d:
        result = calculate_grid_2d(rows, cols, tile_rows, tile_cols)
        match = result == expected
        print(f"  Matrix={rows}x{cols}, Tile={tile_rows}x{tile_cols}: Expected {expected}, Got {result} - {'✓' if match else '✗'}")
    
    print("\n" + "=" * 60)
    print("Run test.py for comprehensive GPU kernel tests!")
    print("=" * 60)
