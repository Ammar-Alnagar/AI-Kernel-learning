# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 02: Kernel Basics - Solution

Reference implementation for kernel decorator, launch, and block IDs.
"""

import cuda.tile as ct
import torch
from math import ceil

ConstInt = ct.Constant[int]
TILE_SIZE = 32


# =============================================================================
# EXERCISE 1: Complete a Basic 1D Kernel
# =============================================================================
@ct.kernel
def square_kernel(input_array, output_array):
    """
    Square each element of the input array.
    """
    block_id = ct.bid(0)
    input_tile = ct.load(input_array, index=(block_id,), shape=(TILE_SIZE,))
    output_tile = input_tile * input_tile
    ct.store(output_array, index=(block_id,), tile=output_tile)


# =============================================================================
# EXERCISE 2: 2D Grid Kernel for Matrix Operations
# =============================================================================
@ct.kernel
def matrix_add_kernel(matrix_a, matrix_b, output_matrix,
                      tile_m: ConstInt, tile_n: ConstInt):
    """
    Add two matrices using a 2D grid.
    """
    block_id_m = ct.bid(0)
    block_id_n = ct.bid(1)
    
    tile_a = ct.load(matrix_a, index=(block_id_m, block_id_n), 
                     shape=(tile_m, tile_n))
    tile_b = ct.load(matrix_b, index=(block_id_m, block_id_n), 
                     shape=(tile_m, tile_n))
    
    result_tile = tile_a + tile_b
    
    ct.store(output_matrix, index=(block_id_m, block_id_n), tile=result_tile)


# =============================================================================
# EXERCISE 3: Calculate Grid Size for Kernel Launch
# =============================================================================
def calculate_grid_1d(array_size: int, tile_size: int) -> tuple:
    """
    Calculate 1D grid size for processing an array.
    """
    # Ceiling division: (n + d - 1) // d
    num_blocks = (array_size + tile_size - 1) // tile_size
    grid = (num_blocks, 1, 1)
    return grid


# =============================================================================
# EXERCISE 4: Calculate 2D Grid Size for Matrix Processing
# =============================================================================
def calculate_grid_2d(rows: int, cols: int, 
                      tile_rows: int, tile_cols: int) -> tuple:
    """
    Calculate 2D grid size for processing a matrix.
    """
    num_blocks_m = (rows + tile_rows - 1) // tile_rows
    num_blocks_n = (cols + tile_cols - 1) // tile_cols
    grid = (num_blocks_m, num_blocks_n, 1)
    return grid


# =============================================================================
# EXERCISE 5: Host Function to Launch a Kernel
# =============================================================================
def scale_and_add(array_a: torch.Tensor, array_b: torch.Tensor, 
                  scale: float) -> torch.Tensor:
    """
    Compute: result = array_a * scale + array_b
    """
    # Validate inputs
    if array_a.shape != array_b.shape:
        raise ValueError("Input arrays must have the same shape")
    if not array_a.is_cuda or not array_b.is_cuda:
        raise ValueError("Input arrays must be on CUDA device")
    
    # Create output tensor
    result = torch.empty_like(array_a)
    
    # Calculate grid size
    grid = calculate_grid_1d(array_a.shape[0], TILE_SIZE)
    
    # Launch kernel
    ct.launch(torch.cuda.current_stream(), grid, scale_add_kernel,
              (array_a, array_b, result, scale))
    
    return result


@ct.kernel
def scale_add_kernel(array_a, array_b, result, scale: float):
    """
    Kernel for computing result = array_a * scale + array_b.
    """
    block_id = ct.bid(0)
    
    tile_a = ct.load(array_a, index=(block_id,), shape=(TILE_SIZE,))
    tile_b = ct.load(array_b, index=(block_id,), shape=(TILE_SIZE,))
    
    result_tile = tile_a * scale + tile_b
    
    ct.store(result, index=(block_id,), tile=result_tile)


# =============================================================================
# EXERCISE 6: 3D Grid for Batched Operations
# =============================================================================
@ct.kernel
def batch_scale_kernel(batch_array, output_array,
                       batch_size: ConstInt, tile_size: ConstInt,
                       scale: float):
    """
    Scale elements in a batched array using 3D grid.
    """
    batch_id = ct.bid(0)
    tile_id = ct.bid(1)
    
    # Load tile with shape (1, tile_size)
    input_tile = ct.load(batch_array, index=(batch_id, tile_id), 
                         shape=(1, tile_size))
    
    # Reshape to 1D for easier processing
    input_tile_1d = ct.reshape(input_tile, (tile_size,))
    
    # Apply scaling
    output_tile_1d = input_tile_1d * scale
    
    # Reshape back to 2D for storing
    output_tile = ct.reshape(output_tile_1d, (1, tile_size))
    
    ct.store(output_array, index=(batch_id, tile_id), tile=output_tile)


# =============================================================================
# EXERCISE 7: Using ct.num_blocks() for Adaptive Processing
# =============================================================================
@ct.kernel
def adaptive_kernel(input_array, output_array, scale: float):
    """
    Process array using ct.num_blocks() for bounds checking.
    """
    block_id = ct.bid(0)
    total_blocks = ct.num_blocks(0)
    
    # Process if within valid range
    if block_id < total_blocks:
        input_tile = ct.load(input_array, index=(block_id,), 
                            shape=(TILE_SIZE,),
                            padding_mode=ct.PaddingMode.ZERO)
        
        output_tile = input_tile * scale
        
        ct.store(output_array, index=(block_id,), tile=output_tile)


# =============================================================================
# Main: Test Implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Module 02: Solution Self-Test")
    print("=" * 60)
    
    # Test grid calculations
    print("\n[Test 1] Grid Calculations")
    print("-" * 40)
    
    print("1D Grid:")
    for size, tile in [(128, 32), (100, 32), (256, 64)]:
        grid = calculate_grid_1d(size, tile)
        print(f"  Size={size}, Tile={tile} → Grid={grid}")
    
    print("2D Grid:")
    for (r, c), (tr, tc) in [((64, 64), (32, 32)), ((100, 50), (32, 32))]:
        grid = calculate_grid_2d(r, c, tr, tc)
        print(f"  Matrix={r}x{c}, Tile={tr}x{tc} → Grid={grid}")
    
    # Quick GPU test
    print("\n[Test 2] Quick GPU Test")
    print("-" * 40)
    
    try:
        a = torch.randn(128, dtype=torch.float32, device='cuda')
        b = torch.randn(128, dtype=torch.float32, device='cuda')
        result = scale_and_add(a, b, 2.0)
        expected = a * 2.0 + b
        match = torch.allclose(result, expected, rtol=1e-5, atol=1e-7)
        print(f"  scale_and_add test: {'✓' if match else '✗'}")
    except Exception as e:
        print(f"  GPU test failed: {e}")
    
    print("\n" + "=" * 60)
