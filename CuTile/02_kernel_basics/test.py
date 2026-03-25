# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 02: Kernel Basics - Test Suite

Run this file to test your kernel.py implementation:
    python test.py
"""

import sys
import torch

try:
    from kernel import (
        TILE_SIZE,
        square_kernel, matrix_add_kernel,
        calculate_grid_1d, calculate_grid_2d,
        scale_and_add, scale_add_kernel,
        batch_scale_kernel, adaptive_kernel
    )
except ImportError as e:
    print(f"ERROR: Cannot import from kernel.py: {e}")
    sys.exit(1)

import cuda.tile as ct


def test_calculate_grid_1d():
    """Test 1D grid calculation."""
    print("\n[Test] 1D Grid Calculation")
    print("-" * 40)
    
    test_cases = [
        (128, 32, (4, 1, 1)),
        (100, 32, (4, 1, 1)),  # Not a multiple
        (256, 64, (4, 1, 1)),
        (129, 32, (5, 1, 1)),  # Just over multiple
        (31, 32, (1, 1, 1)),   # Smaller than tile
    ]
    
    all_passed = True
    for size, tile, expected in test_cases:
        result = calculate_grid_1d(size, tile)
        passed = result == expected
        print(f"  Size={size}, Tile={tile}: Expected {expected}, Got {result} - {'✓' if passed else '✗'}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✓ PASSED: All 1D grid calculations correct")
    else:
        print("✗ FAILED: Some 1D grid calculations incorrect")
    
    return all_passed


def test_calculate_grid_2d():
    """Test 2D grid calculation."""
    print("\n[Test] 2D Grid Calculation")
    print("-" * 40)
    
    test_cases = [
        ((64, 64), (32, 32), (2, 2, 1)),
        ((100, 50), (32, 32), (4, 2, 1)),
        ((128, 256), (64, 64), (2, 4, 1)),
        ((33, 33), (32, 32), (2, 2, 1)),
    ]
    
    all_passed = True
    for (rows, cols), (tile_rows, tile_cols), expected in test_cases:
        result = calculate_grid_2d(rows, cols, tile_rows, tile_cols)
        passed = result == expected
        print(f"  Matrix={rows}x{cols}, Tile={tile_rows}x{tile_cols}: Expected {expected}, Got {result} - {'✓' if passed else '✗'}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✓ PASSED: All 2D grid calculations correct")
    else:
        print("✗ FAILED: Some 2D grid calculations incorrect")
    
    return all_passed


def test_square_kernel():
    """Test element-wise squaring kernel."""
    print("\n[Test] Square Kernel")
    print("-" * 40)
    
    try:
        size = 128
        input_array = torch.randn(size, dtype=torch.float32, device='cuda')
        output_array = torch.empty(size, dtype=torch.float32, device='cuda')
        
        grid = calculate_grid_1d(size, TILE_SIZE)
        ct.launch(torch.cuda.current_stream(), grid, square_kernel,
                  (input_array, output_array))
        
        expected = input_array * input_array
        passed = torch.allclose(output_array, expected, rtol=1e-5, atol=1e-7)
        print(f"  Squaring result correct: {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Square kernel works")
        else:
            print("✗ FAILED: Square kernel incorrect")
            print(f"  Max error: {(output_array - expected).abs().max().item()}")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_matrix_add_kernel():
    """Test 2D matrix addition kernel."""
    print("\n[Test] Matrix Add Kernel (2D Grid)")
    print("-" * 40)
    
    try:
        rows, cols = 64, 64
        tile_m, tile_n = 32, 32
        
        matrix_a = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
        matrix_b = torch.randn(rows, cols, dtype=torch.float32, device='cuda')
        output = torch.empty(rows, cols, dtype=torch.float32, device='cuda')
        
        grid = calculate_grid_2d(rows, cols, tile_m, tile_n)
        ct.launch(torch.cuda.current_stream(), grid, matrix_add_kernel,
                  (matrix_a, matrix_b, output, tile_m, tile_n))
        
        expected = matrix_a + matrix_b
        passed = torch.allclose(output, expected, rtol=1e-5, atol=1e-7)
        print(f"  Matrix addition correct: {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Matrix add kernel works")
        else:
            print("✗ FAILED: Matrix add kernel incorrect")
            print(f"  Max error: {(output - expected).abs().max().item()}")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_scale_and_add():
    """Test combined scale and add operation."""
    print("\n[Test] Scale and Add Function")
    print("-" * 40)
    
    try:
        size = 128
        array_a = torch.randn(size, dtype=torch.float32, device='cuda')
        array_b = torch.randn(size, dtype=torch.float32, device='cuda')
        scale = 2.5
        
        result = scale_and_add(array_a, array_b, scale)
        expected = array_a * scale + array_b
        
        passed = torch.allclose(result, expected, rtol=1e-5, atol=1e-7)
        print(f"  Scale and add result correct: {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Scale and add works")
        else:
            print("✗ FAILED: Scale and add incorrect")
            print(f"  Max error: {(result - expected).abs().max().item()}")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_batch_scale_kernel():
    """Test batched scaling with 3D grid."""
    print("\n[Test] Batch Scale Kernel (3D Grid)")
    print("-" * 40)
    
    try:
        batch_size = 4
        array_size = 128
        tile_size = 32
        
        batch_array = torch.randn(batch_size, array_size, 
                                   dtype=torch.float32, device='cuda')
        output_array = torch.empty(batch_size, array_size, 
                                    dtype=torch.float32, device='cuda')
        scale = 3.0
        
        # 3D grid: (batch, tiles_per_batch, 1)
        num_tiles = (array_size + tile_size - 1) // tile_size
        grid = (batch_size, num_tiles, 1)
        
        ct.launch(torch.cuda.current_stream(), grid, batch_scale_kernel,
                  (batch_array, output_array, batch_size, tile_size, scale))
        
        expected = batch_array * scale
        passed = torch.allclose(output_array, expected, rtol=1e-5, atol=1e-7)
        print(f"  Batch scaling correct: {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Batch scale kernel works")
        else:
            print("✗ FAILED: Batch scale kernel incorrect")
            print(f"  Max error: {(output_array - expected).abs().max().item()}")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_adaptive_kernel():
    """Test adaptive kernel with num_blocks."""
    print("\n[Test] Adaptive Kernel")
    print("-" * 40)
    
    try:
        size = 100  # Not a multiple of TILE_SIZE (32)
        input_array = torch.randn(size, dtype=torch.float32, device='cuda')
        output_array = torch.empty(size, dtype=torch.float32, device='cuda')
        scale = 1.5
        
        grid = calculate_grid_1d(size, TILE_SIZE)
        ct.launch(torch.cuda.current_stream(), grid, adaptive_kernel,
                  (input_array, output_array, scale))
        
        expected = input_array * scale
        passed = torch.allclose(output_array, expected, rtol=1e-5, atol=1e-7)
        print(f"  Adaptive processing correct: {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Adaptive kernel works")
        else:
            print("✗ FAILED: Adaptive kernel incorrect")
            print(f"  Max error: {(output_array - expected).abs().max().item()}")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_non_multiple_tile_size():
    """Test kernels with array sizes not multiple of tile size."""
    print("\n[Test] Non-multiple Tile Size Handling")
    print("-" * 40)
    
    try:
        size = 100  # Not a multiple of 32
        input_array = torch.randn(size, dtype=torch.float32, device='cuda')
        output_array = torch.empty(size, dtype=torch.float32, device='cuda')
        
        grid = calculate_grid_1d(size, TILE_SIZE)
        ct.launch(torch.cuda.current_stream(), grid, square_kernel,
                  (input_array, output_array),
                  padding_mode=ct.PaddingMode.ZERO)
        
        expected = input_array * input_array
        passed = torch.allclose(output_array, expected, rtol=1e-4, atol=1e-5)
        print(f"  Non-multiple size handling: {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Non-multiple tile size handled")
        else:
            print("✗ FAILED: Non-multiple tile size not handled correctly")
            print(f"  Max error: {(output_array - expected).abs().max().item()}")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Module 02: Comprehensive Test Suite")
    print("=" * 60)
    print(f"TILE_SIZE = {TILE_SIZE}")
    
    tests = [
        test_calculate_grid_1d,
        test_calculate_grid_2d,
        test_square_kernel,
        test_matrix_add_kernel,
        test_scale_and_add,
        test_batch_scale_kernel,
        test_adaptive_kernel,
        test_non_multiple_tile_size,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ Test {test.__name__} raised exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n🎉 Congratulations! All tests passed!")
        print("You're ready to move on to Module 03!")
    else:
        print("\n⚠️  Some tests failed. Review your code and try again.")
        print("Check solution.py for reference implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
