# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 01: Data Model - Arrays, Tiles, and Memory Layouts

INSTRUCTIONS:
1. Read the README.md for concepts about arrays vs tiles, strides, and dtypes
2. Fill in the code where you see "# FILL IN:" comments
3. Run test.py to verify your solution
4. Compare with solution.py after completing

KEY CONCEPTS:
- Arrays: Mutable, global memory, runtime shape
- Tiles: Immutable, no storage, compile-time shape (power of 2!)
- Strides: Map logical indices to physical memory
"""

import cuda.tile as ct
import torch

# =============================================================================
# EXERCISE 1: Valid Tile Shapes
# =============================================================================
# Define valid tile shapes (must be powers of 2!)

# FILL IN: Define a valid 1D tile shape (power of 2)
TILE_1D_SIZE =  # FILL IN: Choose 16, 32, 64, or 128

# FILL IN: Define valid 2D tile shapes (both dimensions power of 2)
TILE_2D_M =  # FILL IN: Choose power of 2 (e.g., 32, 64, 128)
TILE_2D_N =  # FILL IN: Choose power of 2 (e.g., 32, 64, 128)

# FILL IN: Define valid 3D tile shapes (all dimensions power of 2)
TILE_3D_X =  # FILL IN: Choose power of 2
TILE_3D_Y =  # FILL IN: Choose power of 2
TILE_3D_Z =  # FILL IN: Choose power of 2


# =============================================================================
# EXERCISE 2: Query Array Properties
# =============================================================================
@ct.kernel
def query_array_properties(input_array, output_array):
    """
    Query and store array properties.
    
    Arrays have runtime-determined properties.
    
    Args:
        input_array: Input array to query
        output_array: Output array to store results
    """
    block_id = ct.bid(0)
    
    # FILL IN: Get the number of dimensions of input_array
    # Hint: Use len(array.shape)
    ndim =  # FILL IN: len(?)
    
    # FILL IN: Get the shape of input_array
    # Hint: array.shape returns a tuple
    shape =  # FILL IN: input_array.?
    
    # FILL IN: Get the dtype of input_array
    # Hint: array.dtype
    dtype =  # FILL IN: input_array.?
    
    # Store results (each block stores one property)
    if block_id == 0:
        # Store ndim as float for compatibility
        output_array[0] = float(ndim)
    elif block_id == 1:
        # Store first dimension size
        # Note: In tile code, we access shape elements differently
        # For this exercise, we'll use a simplified approach
        pass
    elif block_id == 2:
        # dtype info
        pass


# =============================================================================
# EXERCISE 3: Query Tile Properties
# =============================================================================
@ct.kernel
def query_tile_properties(input_array, output_array):
    """
    Query tile properties after loading.
    
    Tiles have compile-time constant properties.
    
    Args:
        input_array: Input array to load from
        output_array: Output array to store results
    """
    block_id = ct.bid(0)
    
    # FILL IN: Load a tile with shape (32,)
    # Hint: ct.load(array, index, shape)
    tile =  # FILL IN: ct.load(?, index=(?,), shape=(?,))
    
    # FILL IN: Get the tile's shape (compile-time constant!)
    # Hint: tile.shape
    tile_shape =  # FILL IN: tile.?
    
    # FILL IN: Get the tile's dtype (compile-time constant!)
    # Hint: tile.dtype
    tile_dtype =  # FILL IN: tile.?
    
    # Store the first dimension of tile shape
    # Note: tile_shape is a compile-time tuple
    # We'll store the size of the first dimension
    # FILL IN: Access first element of tile_shape
    first_dim =  # FILL IN: tile_shape[0]
    
    # Store result
    output_array[block_id] = float(first_dim)


# =============================================================================
# EXERCISE 4: Calculate Memory Address from Strides
# =============================================================================
def calculate_stride_address(base_addr: int, element_size: int, 
                             strides: tuple, indices: tuple) -> int:
    """
    Calculate the memory address for a given index in a strided array.
    
    This is a HOST function (standard Python) that demonstrates stride calculation.
    
    Formula: address = base_addr + element_size * sum(stride[i] * index[i] for all i)
    
    Args:
        base_addr: Base memory address of the array
        element_size: Size of each element in bytes
        strides: Tuple of strides for each dimension
        indices: Tuple of indices for each dimension
    
    Returns:
        Memory address of the element
    """
    # FILL IN: Calculate the offset using strides and indices
    # offset = sum(stride[i] * index[i] for i in range(len(strides)))
    offset =  # FILL IN: Calculate offset
    
    # FILL IN: Calculate final address
    # address = base_addr + element_size * offset
    address =  # FILL IN: Calculate address
    
    return address


# =============================================================================
# EXERCISE 5: Stride Calculation for Row-Major Layout
# =============================================================================
def row_major_strides(shape: tuple) -> tuple:
    """
    Calculate row-major strides for a given shape.
    
    In row-major order:
    - Last dimension has stride 1
    - Each preceding dimension has stride = product of all following dimensions
    
    Example: shape (4, 5, 3) → strides (15, 3, 1)
    
    Args:
        shape: The shape of the array
    
    Returns:
        Tuple of strides in row-major order
    """
    ndim = len(shape)
    strides = [0] * ndim
    
    # FILL IN: Calculate row-major strides
    # Start from the last dimension (stride = 1)
    # Work backwards, multiplying by subsequent dimensions
    stride = 1
    for i in range(ndim - 1, -1, -1):
        strides[i] =  # FILL IN: Set current stride
        stride =  # FILL IN: Update stride for next dimension
    
    return tuple(strides)


# =============================================================================
# EXERCISE 6: Dtype Conversion Kernel
# =============================================================================
@ct.kernel
def convert_dtype_kernel(input_array, output_array):
    """
    Load a tile, convert dtype, and store.
    
    Demonstrates dtype handling in CuTile.
    
    Args:
        input_array: Input array (float32)
        output_array: Output array (will be float16)
    """
    block_id = ct.bid(0)
    
    # FILL IN: Load a tile from input_array with shape (32,)
    input_tile =  # FILL IN: ct.load(?, index=(?,), shape=(?,))
    
    # FILL IN: Convert the tile to float16
    # Hint: Use .astype(ct.float16) or ct.astype(tile, ct.float16)
    output_tile =  # FILL IN: Convert dtype
    
    # FILL IN: Store the converted tile
    # FILL IN: ct.store(?, index=(?,), tile=?)
    pass  # FILL IN: Replace with ct.store call


# =============================================================================
# EXERCISE 7: Tile Arithmetic with Broadcasting
# =============================================================================
@ct.kernel
def tile_broadcast_kernel(input_array, scalar_val, output_array):
    """
    Demonstrate broadcasting: add a scalar to each element of a tile.
    
    Broadcasting extends smaller operands to match larger ones.
    A scalar (0D tile) can be added to a 1D tile.
    
    Args:
        input_array: Input array
        scalar_val: A scalar value (will be broadcast)
        output_array: Output array
    """
    block_id = ct.bid(0)
    
    # FILL IN: Load a tile with shape (64,)
    input_tile =  # FILL IN: Load tile
    
    # FILL IN: Add scalar_val to the tile
    # The scalar is automatically broadcast to match the tile shape
    result_tile =  # FILL IN: input_tile + scalar_val
    
    # FILL IN: Store the result
    # FILL IN: Store result_tile
    pass


# =============================================================================
# EXERCISE 8: Create Tiles with Factory Functions
# =============================================================================
@ct.kernel
def factory_functions_kernel(output_array):
    """
    Practice using tile factory functions.
    
    CuTile provides:
    - ct.zeros(shape) - Create tile filled with zeros
    - ct.ones(shape) - Create tile filled with ones
    - ct.full(shape, value) - Create tile filled with given value
    - ct.arange(size) - Create tile with [0, 1, 2, ..., size-1]
    
    Args:
        output_array: Output array to store results
    """
    block_id = ct.bid(0)
    
    if block_id == 0:
        # FILL IN: Create a tile of zeros with shape (16,)
        # Hint: ct.zeros(shape)
        zeros_tile =  # FILL IN: ct.zeros(?)
        ct.store(output_array, index=(block_id,), tile=zeros_tile)
        
    elif block_id == 1:
        # FILL IN: Create a tile of ones with shape (16,)
        # Hint: ct.ones(shape)
        ones_tile =  # FILL IN: ct.ones(?)
        ct.store(output_array, index=(block_id,), tile=ones_tile)
        
    elif block_id == 2:
        # FILL IN: Create a tile filled with value 42.0, shape (16,)
        # Hint: ct.full(shape, value)
        full_tile =  # FILL IN: ct.full(?, ?)
        ct.store(output_array, index=(block_id,), tile=full_tile)
        
    elif block_id == 3:
        # FILL IN: Create a tile with arange [0, 1, 2, ..., 15]
        # Hint: ct.arange(size)
        arange_tile =  # FILL IN: ct.arange(?)
        ct.store(output_array, index=(block_id,), tile=arange_tile)


# =============================================================================
# Main: Test Your Implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Module 01: Testing Your Data Model Implementation")
    print("=" * 60)
    
    # Test tile sizes are powers of 2
    print("\n[Test 1] Valid Tile Sizes")
    print("-" * 40)
    
    def is_power_of_2(n):
        return n > 0 and (n & (n - 1)) == 0
    
    print(f"TILE_1D_SIZE = {TILE_1D_SIZE}: {'✓' if is_power_of_2(TILE_1D_SIZE) else '✗'}")
    print(f"TILE_2D_M = {TILE_2D_M}, TILE_2D_N = {TILE_2D_N}: ", end="")
    print(f"{'✓' if is_power_of_2(TILE_2D_M) and is_power_of_2(TILE_2D_N) else '✗'}")
    print(f"TILE_3D_X = {TILE_3D_X}, Y = {TILE_3D_Y}, Z = {TILE_3D_Z}: ", end="")
    print(f"{'✓' if all(is_power_of_2(x) for x in [TILE_3D_X, TILE_3D_Y, TILE_3D_Z]) else '✗'}")
    
    # Test stride calculation
    print("\n[Test 2] Stride Calculation")
    print("-" * 40)
    
    # Row-major strides for shape (4, 5, 3) should be (15, 3, 1)
    test_shape = (4, 5, 3)
    expected_strides = (15, 3, 1)
    calculated_strides = row_major_strides(test_shape)
    print(f"Shape: {test_shape}")
    print(f"Expected strides: {expected_strides}")
    print(f"Calculated strides: {calculated_strides}")
    print(f"Match: {'✓' if calculated_strides == expected_strides else '✗'}")
    
    # Test memory address calculation
    print("\n[Test 3] Memory Address Calculation")
    print("-" * 40)
    
    base = 1000
    elem_size = 4  # float32
    strides = (20, 4, 1)
    indices = (2, 3, 1)
    
    addr = calculate_stride_address(base, elem_size, strides, indices)
    expected_addr = base + 4 * (20*2 + 4*3 + 1*1)  # 1000 + 4 * 53 = 1212
    
    print(f"Base: {base}, Element size: {elem_size}")
    print(f"Strides: {strides}, Indices: {indices}")
    print(f"Calculated address: {addr}")
    print(f"Expected address: {expected_addr}")
    print(f"Match: {'✓' if addr == expected_addr else '✗'}")
    
    print("\n" + "=" * 60)
    print("Run test.py for comprehensive GPU kernel tests!")
    print("=" * 60)
