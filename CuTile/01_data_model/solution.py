# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 01: Data Model - Solution

Reference implementation for array/tile properties, strides, and dtypes.
"""

import cuda.tile as ct
import torch

# =============================================================================
# EXERCISE 1: Valid Tile Shapes
# =============================================================================
TILE_1D_SIZE = 32  # 32 = 2^5

TILE_2D_M = 64  # 64 = 2^6
TILE_2D_N = 64  # 64 = 2^6

TILE_3D_X = 8   # 8 = 2^3
TILE_3D_Y = 8   # 8 = 2^3
TILE_3D_Z = 16  # 16 = 2^4


# =============================================================================
# EXERCISE 2: Query Array Properties
# =============================================================================
@ct.kernel
def query_array_properties(input_array, output_array):
    """
    Query and store array properties.
    
    Arrays have runtime-determined properties.
    """
    block_id = ct.bid(0)
    
    # Get the number of dimensions
    ndim = len(input_array.shape)
    
    # Get the shape (tuple of int32 scalars)
    shape = input_array.shape
    
    # Get the dtype
    dtype = input_array.dtype
    
    # Store results
    if block_id == 0:
        output_array[0] = float(ndim)
    elif block_id == 1:
        # Access first element of shape tuple
        # Note: shape elements are int32 scalars
        first_dim = shape[0]
        output_array[1] = float(first_dim)
    elif block_id == 2:
        # Store dtype info (this is simplified - actual dtype handling is more complex)
        output_array[2] = 1.0  # Placeholder


# =============================================================================
# EXERCISE 3: Query Tile Properties
# =============================================================================
@ct.kernel
def query_tile_properties(input_array, output_array):
    """
    Query tile properties after loading.
    
    Tiles have compile-time constant properties.
    """
    block_id = ct.bid(0)
    
    # Load a tile with shape (32,)
    tile = ct.load(input_array, index=(block_id,), shape=(32,))
    
    # Get the tile's shape (compile-time constant!)
    tile_shape = tile.shape  # (32,) - known at compile time
    
    # Get the tile's dtype (compile-time constant!)
    tile_dtype = tile.dtype  # e.g., ct.float32
    
    # Access first element of tile_shape
    first_dim = tile_shape[0]  # 32 - compile-time constant
    
    # Store result
    output_array[block_id] = float(first_dim)


# =============================================================================
# EXERCISE 4: Calculate Memory Address from Strides
# =============================================================================
def calculate_stride_address(base_addr: int, element_size: int, 
                             strides: tuple, indices: tuple) -> int:
    """
    Calculate the memory address for a given index in a strided array.
    """
    # Calculate the offset using strides and indices
    offset = sum(s * i for s, i in zip(strides, indices))
    
    # Calculate final address
    address = base_addr + element_size * offset
    
    return address


# =============================================================================
# EXERCISE 5: Stride Calculation for Row-Major Layout
# =============================================================================
def row_major_strides(shape: tuple) -> tuple:
    """
    Calculate row-major strides for a given shape.
    """
    ndim = len(shape)
    strides = [0] * ndim
    
    # Start from the last dimension (stride = 1)
    # Work backwards, multiplying by subsequent dimensions
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] = stride
        stride *= shape[i]
    
    return tuple(strides)


# =============================================================================
# EXERCISE 6: Dtype Conversion Kernel
# =============================================================================
@ct.kernel
def convert_dtype_kernel(input_array, output_array):
    """
    Load a tile, convert dtype, and store.
    """
    block_id = ct.bid(0)
    
    # Load a tile from input_array with shape (32,)
    input_tile = ct.load(input_array, index=(block_id,), shape=(32,))
    
    # Convert the tile to float16
    output_tile = input_tile.astype(ct.float16)
    
    # Store the converted tile
    ct.store(output_array, index=(block_id,), tile=output_tile)


# =============================================================================
# EXERCISE 7: Tile Arithmetic with Broadcasting
# =============================================================================
@ct.kernel
def tile_broadcast_kernel(input_array, scalar_val, output_array):
    """
    Demonstrate broadcasting: add a scalar to each element of a tile.
    """
    block_id = ct.bid(0)
    
    # Load a tile with shape (64,)
    input_tile = ct.load(input_array, index=(block_id,), shape=(64,))
    
    # Add scalar_val to the tile (automatically broadcast)
    result_tile = input_tile + scalar_val
    
    # Store the result
    ct.store(output_array, index=(block_id,), tile=result_tile)


# =============================================================================
# EXERCISE 8: Create Tiles with Factory Functions
# =============================================================================
@ct.kernel
def factory_functions_kernel(output_array):
    """
    Practice using tile factory functions.
    """
    block_id = ct.bid(0)
    
    if block_id == 0:
        # Create a tile of zeros with shape (16,)
        zeros_tile = ct.zeros((16,))
        ct.store(output_array, index=(block_id,), tile=zeros_tile)
        
    elif block_id == 1:
        # Create a tile of ones with shape (16,)
        ones_tile = ct.ones((16,))
        ct.store(output_array, index=(block_id,), tile=ones_tile)
        
    elif block_id == 2:
        # Create a tile filled with value 42.0, shape (16,)
        full_tile = ct.full((16,), 42.0)
        ct.store(output_array, index=(block_id,), tile=full_tile)
        
    elif block_id == 3:
        # Create a tile with arange [0, 1, 2, ..., 15]
        arange_tile = ct.arange(16)
        ct.store(output_array, index=(block_id,), tile=arange_tile)


# =============================================================================
# Main: Test Implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Module 01: Solution Self-Test")
    print("=" * 60)
    
    # Test tile sizes
    print("\n[Test 1] Tile Sizes")
    print("-" * 40)
    print(f"TILE_1D_SIZE = {TILE_1D_SIZE}")
    print(f"TILE_2D_M = {TILE_2D_M}, TILE_2D_N = {TILE_2D_N}")
    print(f"TILE_3D_X = {TILE_3D_X}, Y = {TILE_3D_Y}, Z = {TILE_3D_Z}")
    
    # Test stride calculation
    print("\n[Test 2] Row-Major Strides")
    print("-" * 40)
    test_shape = (4, 5, 3)
    expected = (15, 3, 1)
    result = row_major_strides(test_shape)
    print(f"Shape: {test_shape}")
    print(f"Expected: {expected}, Got: {result}")
    print(f"Match: {result == expected}")
    
    # Test address calculation
    print("\n[Test 3] Memory Address Calculation")
    print("-" * 40)
    base = 1000
    elem_size = 4
    strides = (20, 4, 1)
    indices = (2, 3, 1)
    addr = calculate_stride_address(base, elem_size, strides, indices)
    expected_addr = 1212
    print(f"Address: {addr}, Expected: {expected_addr}")
    print(f"Match: {addr == expected_addr}")
    
    print("\n" + "=" * 60)
