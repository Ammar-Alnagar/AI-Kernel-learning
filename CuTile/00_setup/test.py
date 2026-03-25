# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 00: Setup & First Kernel - Test Suite

Run this file to test your kernel.py implementation:
    python test.py

This test suite verifies:
1. Vector addition correctness
2. Scalar multiplication correctness
3. Edge cases (different sizes, dtypes)
4. Error handling
"""

import sys
import torch

# Import the student's implementation
try:
    from kernel import vector_add, scalar_multiply, TILE_SIZE
except ImportError as e:
    print(f"ERROR: Cannot import from kernel.py: {e}")
    print("Make sure kernel.py is in the same directory as test.py")
    sys.exit(1)


def test_tile_size_is_power_of_2():
    """Test that TILE_SIZE is a valid power of 2."""
    print("\n[Test] TILE_SIZE is power of 2")
    print("-" * 40)
    
    valid_sizes = [16, 32, 64, 128, 256]
    
    if TILE_SIZE not in valid_sizes:
        print(f"✗ FAILED: TILE_SIZE={TILE_SIZE} should be a power of 2")
        print(f"  Valid sizes: {valid_sizes}")
        return False
    
    print(f"✓ PASSED: TILE_SIZE={TILE_SIZE} is valid")
    return True


def test_vector_add_correctness():
    """Test that vector addition produces correct results."""
    print("\n[Test] Vector Addition Correctness")
    print("-" * 40)
    
    # Test with various sizes
    test_sizes = [64, 128, 256, 512]
    
    for size in test_sizes:
        a = torch.randn(size, dtype=torch.float32, device='cuda')
        b = torch.randn(size, dtype=torch.float32, device='cuda')
        
        result = vector_add(a, b)
        expected = a + b
        
        if not torch.allclose(result, expected, rtol=1e-5, atol=1e-7):
            print(f"✗ FAILED: Size {size}")
            print(f"  Max error: {(result - expected).abs().max().item()}")
            return False
    
    print(f"✓ PASSED: All sizes tested successfully")
    return True


def test_vector_add_dtypes():
    """Test vector addition with different data types."""
    print("\n[Test] Vector Addition Data Types")
    print("-" * 40)
    
    dtypes = [torch.float16, torch.float32, torch.bfloat16]
    size = 128
    
    for dtype in dtypes:
        try:
            a = torch.randn(size, dtype=dtype, device='cuda')
            b = torch.randn(size, dtype=dtype, device='cuda')
            
            result = vector_add(a, b)
            expected = a + b
            
            # Use looser tolerance for float16
            if dtype == torch.float16:
                atol, rtol = 1e-2, 1e-2
            else:
                atol, rtol = 1e-5, 1e-7
            
            if not torch.allclose(result, expected, rtol=rtol, atol=atol):
                print(f"✗ FAILED: dtype={dtype}")
                return False
        except Exception as e:
            print(f"✗ FAILED: dtype={dtype} raised {e}")
            return False
    
    print(f"✓ PASSED: All dtypes tested successfully")
    return True


def test_scalar_multiply_correctness():
    """Test that scalar multiplication produces correct results."""
    print("\n[Test] Scalar Multiplication Correctness")
    print("-" * 40)
    
    test_cases = [
        (64, 2.0),
        (128, 0.5),
        (256, -1.5),
        (512, 3.14159),
    ]
    
    for size, scalar in test_cases:
        a = torch.randn(size, dtype=torch.float32, device='cuda')
        
        result = scalar_multiply(a, scalar)
        expected = a * scalar
        
        if not torch.allclose(result, expected, rtol=1e-5, atol=1e-7):
            print(f"✗ FAILED: Size {size}, Scalar {scalar}")
            print(f"  Max error: {(result - expected).abs().max().item()}")
            return False
    
    print(f"✓ PASSED: All scalar multiplication tests passed")
    return True


def test_scalar_multiply_edge_cases():
    """Test scalar multiplication with edge case scalars."""
    print("\n[Test] Scalar Multiplication Edge Cases")
    print("-" * 40)
    
    size = 128
    a = torch.ones(size, dtype=torch.float32, device='cuda')
    
    edge_cases = [
        (0.0, "zero"),
        (1.0, "identity"),
        (-1.0, "negative one"),
    ]
    
    for scalar, name in edge_cases:
        result = scalar_multiply(a, scalar)
        expected = a * scalar
        
        if not torch.allclose(result, expected, rtol=1e-5, atol=1e-7):
            print(f"✗ FAILED: {name} (scalar={scalar})")
            return False
    
    print(f"✓ PASSED: All edge cases handled correctly")
    return True


def test_non_multiple_tile_size():
    """Test when array size is not a multiple of tile size."""
    print("\n[Test] Non-multiple Tile Size")
    print("-" * 40)
    
    # Size that's not a multiple of TILE_SIZE
    size = 100  # If TILE_SIZE=16, this is 6.25 tiles
    
    a = torch.randn(size, dtype=torch.float32, device='cuda')
    b = torch.randn(size, dtype=torch.float32, device='cuda')
    
    try:
        result = vector_add(a, b)
        expected = a + b
        
        if not torch.allclose(result, expected, rtol=1e-5, atol=1e-7):
            print(f"✗ FAILED: Non-multiple size {size}")
            print(f"  Max error: {(result - expected).abs().max().item()}")
            return False
        
        print(f"✓ PASSED: Non-multiple size handled correctly")
        return True
    except Exception as e:
        print(f"✗ FAILED: Exception raised: {e}")
        return False


def run_all_tests():
    """Run all tests and report results."""
    print("=" * 60)
    print("Module 00: Comprehensive Test Suite")
    print("=" * 60)
    print(f"TILE_SIZE = {TILE_SIZE}")
    
    tests = [
        test_tile_size_is_power_of_2,
        test_vector_add_correctness,
        test_vector_add_dtypes,
        test_scalar_multiply_correctness,
        test_scalar_multiply_edge_cases,
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
        print("You're ready to move on to Module 01!")
    else:
        print("\n⚠️  Some tests failed. Review your code and try again.")
        print("Check solution.py for reference implementation.")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
