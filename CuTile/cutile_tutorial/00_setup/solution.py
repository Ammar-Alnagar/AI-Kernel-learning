# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 00: Setup & First Kernel - Solution

This is the reference implementation. Compare with your code after completing the exercises!
"""

import cuda.tile as ct
import torch

# =============================================================================
# EXERCISE 1: Define a Tile Size Constant
# =============================================================================
TILE_SIZE = 16  # Each tile processes 16 elements (power of 2)


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
    # Get the block ID along dimension 0
    block_id = ct.bid(0)
    
    # Load tiles from global memory
    # ct.load(array, index, shape) - index is tile-space position
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    
    # Perform element-wise addition on tiles
    result_tile = a_tile + b_tile
    
    # Store result back to global memory
    ct.store(result, index=(block_id,), tile=result_tile)


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
    
    # Create output tensor with same shape and dtype as inputs
    result = torch.empty_like(a)
    
    # Calculate the grid size using ceiling division
    # Formula: ceil(array_length / TILE_SIZE) = (n + TILE_SIZE - 1) // TILE_SIZE
    grid_size = (a.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    
    # Define the grid as a 3-tuple (x, y, z)
    # We're doing 1D processing, so y and z are 1
    grid = (grid_size, 1, 1)
    
    # Launch the kernel
    ct.launch(torch.cuda.current_stream(), grid, vector_add_kernel, (a, b, result))
    
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
    # Get the block ID
    block_id = ct.bid(0)
    
    # Load a tile from the input array
    input_tile = ct.load(array, index=(block_id,), shape=(TILE_SIZE,))
    
    # Multiply the tile by the scalar
    result_tile = input_tile * scalar
    
    # Store the result tile
    ct.store(result, index=(block_id,), tile=result_tile)


def scalar_multiply(array: torch.Tensor, scalar: float) -> torch.Tensor:
    """
    Host function to multiply an array by a scalar.
    
    Args:
        array: Input tensor (1D, on CUDA)
        scalar: Scalar value to multiply by
    
    Returns:
        result: Output tensor containing array * scalar
    """
    # Create output tensor
    result = torch.empty_like(array)
    
    # Calculate grid size
    grid_size = (array.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    
    # Define grid
    grid = (grid_size, 1, 1)
    
    # Launch the kernel
    ct.launch(torch.cuda.current_stream(), grid, scalar_multiply_kernel, (array, result, scalar))
    
    return result


# =============================================================================
# Main: Test Your Implementation
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Module 00: Solution Self-Test")
    print("=" * 60)
    
    # Test vector addition
    print("\n[Test 1] Vector Addition")
    print("-" * 40)
    
    size = 128
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')
    
    print(f"Input size: {size} elements")
    print(f"Tile size: {TILE_SIZE}")
    
    result = vector_add(a, b)
    expected = a + b
    
    is_correct = torch.allclose(result, expected, rtol=1e-5, atol=1e-7)
    print(f"Results match expected: {is_correct}")
    
    if is_correct:
        print("✓ Vector addition PASSED!")
    else:
        print("✗ Vector addition FAILED!")
        print(f"  Max error: {(result - expected).abs().max().item()}")
    
    # Test scalar multiplication
    print("\n[Test 2] Scalar Multiplication")
    print("-" * 40)
    
    scalar = 2.5
    result = scalar_multiply(a, scalar)
    expected = a * scalar
    
    is_correct = torch.allclose(result, expected, rtol=1e-5, atol=1e-7)
    print(f"Results match expected: {is_correct}")
    
    if is_correct:
        print("✓ Scalar multiplication PASSED!")
    else:
        print("✗ Scalar multiplication FAILED!")
        print(f"  Max error: {(result - expected).abs().max().item()}")
    
    print("\n" + "=" * 60)
    print("All tests complete!")
    print("=" * 60)
