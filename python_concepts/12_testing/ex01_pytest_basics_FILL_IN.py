"""
Module 12 — Testing
Exercise 01 — Pytest Basics

WHAT YOU'RE BUILDING:
  Testing is essential for kernel code:
  - Verify kernel outputs match reference (torch.matmul)
  - Test edge cases (small sizes, large sizes, odd shapes)
  - Catch regressions when optimizing kernels
  
  pytest is the standard Python testing framework.

OBJECTIVE:
  - Write test functions with assertions
  - Use pytest fixtures for setup/teardown
  - Parametrize tests for multiple inputs
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between assert and pytest.assert?
# Q2: Why use fixtures instead of setup/teardown methods?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from typing import Tuple

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Write a simple test function.
#              Test that your matmul produces correct shape.
# HINT: def test_something(): assert condition

def matmul_reference(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Reference matmul implementation."""
    return torch.matmul(a, b)

def test_matmul_output_shape():
    """Test that matmul output has correct shape."""
    # TODO: create test tensors, run matmul, assert shape
    pass

# TODO [EASY]: Test numerical correctness with tolerance.
#              Use torch.allclose for floating point comparison.
# HINT: assert torch.allclose(result, expected, rtol=1e-5, atol=1e-7)

def test_matmul_numerical_correctness():
    """Test matmul produces numerically correct results."""
    # TODO: compare your implementation against reference
    pass

# TODO [MEDIUM]: Use pytest.fixture for test setup.
#              Fixtures are reusable setup code.
# HINT: @pytest.fixture def sample_tensors(): return a, b

# Note: Uncomment import pytest when running with pytest
# import pytest

def create_sample_tensors() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample tensors for testing.
    
    This will be a fixture.
    """
    a = torch.randn(10, 20)
    b = torch.randn(20, 30)
    return a, b

# TODO: Convert create_sample_tensors to pytest fixture
# @pytest.fixture
# def sample_tensors():
#     return create_sample_tensors()

# TODO [MEDIUM]: Use pytest.mark.parametrize for multiple test cases.
#              Test multiple matrix sizes with one test function.
# HINT: @pytest.mark.parametrize("M,N,K", [(10, 20, 30), (100, 200, 300), ...])

def test_matmul_multiple_sizes():
    """Test matmul with multiple matrix sizes."""
    # TODO: add parametrize decorator with multiple sizes
    # Test function should verify output shape for each size
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How do you run pytest on this file?
# C2: Why parametrize instead of a loop inside one test?

if __name__ == "__main__":
    # Run tests manually (without pytest)
    print("Running tests manually...\n")
    
    print("test_matmul_output_shape...")
    try:
        test_matmul_output_shape()
        print("  PASSED ✓")
    except AssertionError:
        print("  FAILED ✗")
    
    print("\ntest_matmul_numerical_correctness...")
    try:
        test_matmul_numerical_correctness()
        print("  PASSED ✓")
    except AssertionError:
        print("  FAILED ✗")
    
    print("\nTo run with pytest: pytest ex01_pytest_basics_FILL_IN.py -v")
    print("\nDone!")
