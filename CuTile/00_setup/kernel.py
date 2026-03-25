# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 00: Setup & First Kernel - Fill-in Code Exercise

INSTRUCTIONS:
1. Read the README.md for concepts
2. Fill in the code where you see "# FILL IN:" comments
3. Run test.py to verify your solution
4. Compare with solution.py after completing

TIPS:
- Tile sizes must be powers of 2 (e.g., 16, 32, 64, 128)
- Use ct.bid(0) to get the block ID
- Use ct.load() to load data from arrays into tiles
- Use ct.store() to write tiles back to arrays
"""

import cuda.tile as ct
import torch

# =============================================================================
# EXERCISE 1: Define a Tile Size Constant
# =============================================================================
# Choose a tile size for processing. It must be a power of 2.
# Common choices: 16, 32, 64, 128
# FILL IN: Define TILE_SIZE as a constant integer
TILE_SIZE =  # FILL IN: Choose a power of 2 (e.g., 16, 32, 64)


# =============================================================================
# EXERCISE 2: Complete the Vector Addition Kernel
# =============================================================================
@ct.kernel
def vector_add_kernel(a, b, result):
    """
    Add two vectors element-wise in parallel on the GPU.
    
    Each block processes one tile of the input arrays.
    The block ID determines which portion of data to process.
    
    Args:
        a: Input array 1 (1D tensor)
        b: Input array 2 (1D tensor)
        result: Output array (1D tensor, same shape as inputs)
    """
    # FILL IN: Get the block ID along dimension 0
    # Hint: Use ct.bid() to get the block ID
    block_id =  # FILL IN: ct.bid(?)
    
    # FILL IN: Load a tile from array 'a'
    # Hint: ct.load(array, index, shape)
    # The index should be a tuple with block_id
    # The shape should be a tuple with TILE_SIZE
    a_tile =  # FILL IN: ct.load(?, index=(?,), shape=(?,))
    
    # FILL IN: Load a tile from array 'b' (same pattern as above)
    b_tile =  # FILL IN: ct.load(?, index=(?,), shape=(?,))
    
    # FILL IN: Perform element-wise addition
    # Hint: Simply use the + operator on tiles
    result_tile =  # FILL IN: a_tile ? b_tile
    
    # FILL IN: Store the result tile back to the result array
    # Hint: ct.store(array, index, tile)
    # FILL IN: ct.store(?, index=(?,), tile=?)


# =============================================================================
# EXERCISE 3: Complete the Host Code to Launch the Kernel
# =============================================================================
def vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Host function that launches the vector addition kernel.
    
    Args:
        a: Input tensor 1 (1D, on CUDA device)
        b: Input tensor 2 (1D, on CUDA device, same shape as a)
    
    Returns:
        result: Output tensor containing a + b
    """
    # Validate inputs
    if a.shape != b.shape:
        raise ValueError("Input tensors must have the same shape")
    if a.device != b.device or not a.is_cuda:
        raise ValueError("Input tensors must be on the same CUDA device")
    
    # FILL IN: Create output tensor with same shape and dtype as inputs
    # Hint: Use torch.zeros_like() or torch.empty_like()
    result =  # FILL IN: Create output tensor
    
    # FILL IN: Calculate the grid size
    # We need enough blocks to cover all elements
    # Formula: ceil(array_length / TILE_SIZE)
    # Hint: Use (n + TILE_SIZE - 1) // TILE_SIZE for ceiling division
    grid_size =  # FILL IN: Calculate number of blocks needed
    
    # FILL IN: Define the grid as a 3-tuple (x, y, z)
    # We're doing 1D processing, so y and z are 1
    grid =  # FILL IN: (grid_size, ?, ?)
    
    # FILL IN: Launch the kernel
    # Hint: ct.launch(stream, grid, kernel, args)
    # Use torch.cuda.current_stream() for the stream
    # Pass (a, b, result) as the arguments tuple
    # FILL IN: ct.launch(?, ?, ?, ?)
    
    return result


# =============================================================================
# EXERCISE 4: Create a Scalar Multiplication Kernel (Challenge!)
# =============================================================================
@ct.kernel
def scalar_multiply_kernel(array, result, scalar):
    """
    Multiply each element of an array by a scalar value.
    
    Args:
        array: Input array (1D tensor)
        result: Output array (1D tensor)
        scalar: A constant scalar value (compile-time constant)
    """
    # FILL IN: Get the block ID
    block_id =  # FILL IN: Get block ID
    
    # FILL IN: Load a tile from the input array
    input_tile =  # FILL IN: Load tile
    
    # FILL IN: Multiply the tile by the scalar
    # Hint: Use the * operator
    result_tile =  # FILL IN: input_tile ? scalar
    
    # FILL IN: Store the result tile
    # FILL IN: Store result_tile to result array


def scalar_multiply(array: torch.Tensor, scalar: float) -> torch.Tensor:
    """
    Host function to multiply an array by a scalar.
    
    Args:
        array: Input tensor (1D, on CUDA)
        scalar: Scalar value to multiply by
    
    Returns:
        result: Output tensor containing array * scalar
    """
    # FILL IN: Create output tensor
    result =  # FILL IN: Create output tensor
    
    # FILL IN: Calculate grid size
    grid_size =  # FILL IN: Calculate number of blocks
    
    # FILL IN: Define grid
    grid =  # FILL IN: (grid_size, 1, 1)
    
    # FILL IN: Launch the kernel
    # Note: scalar is passed as a constant argument
    # FILL IN: ct.launch(?, ?, ?, ?)
    
    return result


# =============================================================================
# Main: Test Your Implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Module 00: Testing Your CuTile Implementation")
    print("=" * 60)
    
    # Test vector addition
    print("\n[Test 1] Vector Addition")
    print("-" * 40)
    
    # Create test data
    size = 128
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')
    
    print(f"Input size: {size} elements")
    print(f"Tile size: {TILE_SIZE}")
    
    # Run your kernel
    try:
        result = vector_add(a, b)
        expected = a + b
        
        # Check correctness
        is_correct = torch.allclose(result, expected, rtol=1e-5, atol=1e-7)
        print(f"Results match expected: {is_correct}")
        
        if is_correct:
            print("✓ Vector addition PASSED!")
        else:
            print("✗ Vector addition FAILED!")
            print(f"  Max error: {(result - expected).abs().max().item()}")
    except Exception as e:
        print(f"✗ Vector addition FAILED with error: {e}")
    
    # Test scalar multiplication
    print("\n[Test 2] Scalar Multiplication")
    print("-" * 40)
    
    scalar = 2.5
    try:
        result = scalar_multiply(a, scalar)
        expected = a * scalar
        
        is_correct = torch.allclose(result, expected, rtol=1e-5, atol=1e-7)
        print(f"Results match expected: {is_correct}")
        
        if is_correct:
            print("✓ Scalar multiplication PASSED!")
        else:
            print("✗ Scalar multiplication FAILED!")
            print(f"  Max error: {(result - expected).abs().max().item()}")
    except Exception as e:
        print(f"✗ Scalar multiplication FAILED with error: {e}")
    
    print("\n" + "=" * 60)
    print("Testing complete! Check test.py for more comprehensive tests.")
    print("=" * 60)
