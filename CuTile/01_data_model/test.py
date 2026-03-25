# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 01: Data Model - Test Suite

Run this file to test your kernel.py implementation:
    python test.py
"""

import sys
import torch

try:
    from kernel import (
        TILE_1D_SIZE, TILE_2D_M, TILE_2D_N, TILE_3D_X, TILE_3D_Y, TILE_3D_Z,
        query_array_properties, query_tile_properties,
        convert_dtype_kernel, tile_broadcast_kernel,
        factory_functions_kernel, row_major_strides, calculate_stride_address
    )
except ImportError as e:
    print(f"ERROR: Cannot import from kernel.py: {e}")
    sys.exit(1)

import cuda.tile as ct


def is_power_of_2(n):
    """Check if n is a power of 2."""
    return n > 0 and (n & (n - 1)) == 0


def test_tile_sizes_valid():
    """Test that all tile sizes are powers of 2."""
    print("\n[Test] Tile Sizes are Powers of 2")
    print("-" * 40)
    
    tests = [
        ("TILE_1D_SIZE", TILE_1D_SIZE),
        ("TILE_2D_M", TILE_2D_M),
        ("TILE_2D_N", TILE_2D_N),
        ("TILE_3D_X", TILE_3D_X),
        ("TILE_3D_Y", TILE_3D_Y),
        ("TILE_3D_Z", TILE_3D_Z),
    ]
    
    all_valid = True
    for name, size in tests:
        valid = is_power_of_2(size)
        print(f"  {name} = {size}: {'✓' if valid else '✗'}")
        if not valid:
            all_valid = False
    
    if all_valid:
        print("✓ PASSED: All tile sizes are valid")
    else:
        print("✗ FAILED: Some tile sizes are invalid")
    
    return all_valid


def test_row_major_strides():
    """Test row-major stride calculation."""
    print("\n[Test] Row-Major Strides")
    print("-" * 40)
    
    test_cases = [
        ((4, 5, 3), (15, 3, 1)),
        ((3, 4), (4, 1)),
        ((2, 3, 4, 5), (60, 20, 5, 1)),
        ((10,), (1,)),
    ]
    
    all_passed = True
    for shape, expected in test_cases:
        result = row_major_strides(shape)
        passed = result == expected
        print(f"  Shape {shape}: Expected {expected}, Got {result} - {'✓' if passed else '✗'}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✓ PASSED: All stride calculations correct")
    else:
        print("✗ FAILED: Some stride calculations incorrect")
    
    return all_passed


def test_stride_address_calculation():
    """Test memory address calculation from strides."""
    print("\n[Test] Memory Address Calculation")
    print("-" * 40)
    
    test_cases = [
        (1000, 4, (20, 4, 1), (2, 3, 1), 1212),
        (0, 1, (5, 1), (0, 0), 0),
        (0, 1, (5, 1), (3, 4), 19),
        (500, 8, (10, 1), (5, 3), 1036),
    ]
    
    all_passed = True
    for base, elem_size, strides, indices, expected in test_cases:
        result = calculate_stride_address(base, elem_size, strides, indices)
        passed = result == expected
        print(f"  base={base}, strides={strides}, indices={indices}: Expected {expected}, Got {result} - {'✓' if passed else '✗'}")
        if not passed:
            all_passed = False
    
    if all_passed:
        print("✓ PASSED: All address calculations correct")
    else:
        print("✗ FAILED: Some address calculations incorrect")
    
    return all_passed


def test_query_tile_properties():
    """Test querying tile properties in kernel."""
    print("\n[Test] Query Tile Properties Kernel")
    print("-" * 40)
    
    try:
        # Create input array
        size = 128
        input_array = torch.randn(size, dtype=torch.float32, device='cuda')
        output_array = torch.zeros(size, dtype=torch.float32, device='cuda')
        
        # Launch kernel
        grid = (4, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, query_tile_properties, 
                  (input_array, output_array))
        
        # Check that first block stored correct tile size (32)
        tile_size = output_array[0].item()
        passed = tile_size == 32.0
        print(f"  Tile size stored: {tile_size} (expected 32.0) - {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Tile property query works")
        else:
            print("✗ FAILED: Tile property query incorrect")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_dtype_conversion():
    """Test dtype conversion kernel."""
    print("\n[Test] Dtype Conversion Kernel")
    print("-" * 40)
    
    try:
        # Create input array (float32)
        size = 128
        input_array = torch.randn(size, dtype=torch.float32, device='cuda')
        output_array = torch.empty(size, dtype=torch.float16, device='cuda')
        
        # Launch kernel
        grid = (4, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, convert_dtype_kernel, 
                  (input_array, output_array))
        
        # Check output dtype
        passed = output_array.dtype == torch.float16
        print(f"  Output dtype: {output_array.dtype} - {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Dtype conversion works")
        else:
            print("✗ FAILED: Dtype conversion incorrect")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_broadcasting():
    """Test tile broadcasting kernel."""
    print("\n[Test] Broadcasting Kernel")
    print("-" * 40)
    
    try:
        # Create input array
        size = 256
        input_array = torch.ones(size, dtype=torch.float32, device='cuda')
        output_array = torch.zeros(size, dtype=torch.float32, device='cuda')
        scalar = 5.0
        
        # Launch kernel
        grid = (4, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, tile_broadcast_kernel, 
                  (input_array, scalar, output_array))
        
        # Check result: should be 1 + 5 = 6
        expected = torch.ones(size, dtype=torch.float32, device='cuda') * 6.0
        passed = torch.allclose(output_array, expected, rtol=1e-5, atol=1e-7)
        print(f"  Broadcasting result correct: {'✓' if passed else '✗'}")
        
        if passed:
            print("✓ PASSED: Broadcasting works")
        else:
            print("✗ FAILED: Broadcasting incorrect")
            print(f"  Max error: {(output_array - expected).abs().max().item()}")
        
        return passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def test_factory_functions():
    """Test tile factory functions kernel."""
    print("\n[Test] Factory Functions Kernel")
    print("-" * 40)
    
    try:
        # Create output array (need 4 tiles of 16 elements each)
        output_array = torch.zeros(64, dtype=torch.float32, device='cuda')
        
        # Launch kernel
        grid = (4, 1, 1)
        ct.launch(torch.cuda.current_stream(), grid, factory_functions_kernel, 
                  (output_array,))
        
        # Check each tile
        zeros_tile = output_array[0:16]
        ones_tile = output_array[16:32]
        full_tile = output_array[32:48]
        arange_tile = output_array[48:64]
        
        # Check zeros
        zeros_ok = torch.allclose(zeros_tile, torch.zeros(16, device='cuda'))
        print(f"  zeros tile: {'✓' if zeros_ok else '✗'}")
        
        # Check ones
        ones_ok = torch.allclose(ones_tile, torch.ones(16, device='cuda'))
        print(f"  ones tile: {'✓' if ones_ok else '✗'}")
        
        # Check full (42.0)
        full_ok = torch.allclose(full_tile, torch.full((16,), 42.0, device='cuda'))
        print(f"  full tile (42.0): {'✓' if full_ok else '✗'}")
        
        # Check arange
        expected_arange = torch.arange(16, dtype=torch.float32, device='cuda')
        arange_ok = torch.allclose(arange_tile, expected_arange)
        print(f"  arange tile: {'✓' if arange_ok else '✗'}")
        
        all_passed = zeros_ok and ones_ok and full_ok and arange_ok
        
        if all_passed:
            print("✓ PASSED: All factory functions work")
        else:
            print("✗ FAILED: Some factory functions incorrect")
        
        return all_passed
        
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Module 01: Comprehensive Test Suite")
    print("=" * 60)
    
    tests = [
        test_tile_sizes_valid,
        test_row_major_strides,
        test_stride_address_calculation,
        test_query_tile_properties,
        test_dtype_conversion,
        test_broadcasting,
        test_factory_functions,
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
        print("You're ready to move on to Module 02!")
    else:
        print("\n⚠️  Some tests failed. Review your code and try again.")
        print("Check solution.py for reference implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
