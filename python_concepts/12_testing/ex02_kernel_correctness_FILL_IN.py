"""
Module 12 — Testing
Exercise 02 — Testing Kernel Correctness

WHAT YOU'RE BUILDING:
  Kernel testing requires special patterns:
  - Compare against reference (PyTorch) with tolerance
  - Test edge cases (empty, single element, max size)
  - Test different dtypes (float16, float32, int8)
  
  This is how Triton, CUTLASS, and torch.compile validate correctness.

OBJECTIVE:
  - Test kernel correctness with numerical tolerance
  - Use hypothesis for property-based testing
  - Test across dtypes and devices
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What tolerance (rtol, atol) is appropriate for float16 vs float32?
# Q2: What edge cases matter most for kernel testing?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from typing import List, Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Test kernel with different dtypes.
#              Each dtype has different numerical tolerance.
# HINT: for dtype in [torch.float32, torch.float16]: test with appropriate rtol

def get_tolerance_for_dtype(dtype: torch.dtype) -> Tuple[float, float]:
    """Get (rtol, atol) tolerance for dtype.
    
    Returns:
        (rtol, atol) tuple for torch.allclose
    """
    # TODO: implement tolerance lookup
    # float32: rtol=1e-5, atol=1e-7
    # float16: rtol=1e-3, atol=1e-5
    # int8: exact comparison (rtol=0, atol=0)
    pass

def test_matmul_with_dtype(dtype: torch.dtype):
    """Test matmul correctness for specific dtype."""
    # TODO: create tensors with dtype, run matmul, compare with tolerance
    pass

# TODO [MEDIUM]: Test edge cases.
#              Edge cases catch boundary bugs in kernels.
# HINT: Test empty, single element, very large, non-divisible sizes

def test_matmul_edge_cases():
    """Test matmul with edge case inputs."""
    edge_cases = [
        # (M, N, K)
        (0, 0, 0),       # Empty
        (1, 1, 1),       # Single element
        (1, 128, 128),   # Broadcast-like
        (128, 1, 128),   # Broadcast-like
        (127, 127, 127), # Prime (non-power-of-2)
        (1024, 1024, 1), # K=1 (dot product per output)
    ]
    
    # TODO: test each edge case
    pass

# TODO [HARD]: Property-based testing with hypothesis.
#              Generate random inputs and verify invariants.
#              Install with: pip install hypothesis
# HINT: from hypothesis import given, strategies as st
#       @given(st.integers(1, 100), st.integers(1, 100), st.integers(1, 100))

def test_matmul_properties():
    """Test matmul properties (associativity, etc.).
    
    Properties to verify:
    - (A @ B).shape == (A.shape[0], B.shape[1])
    - A @ I = A (identity)
    - (A @ B).T = B.T @ A.T
    """
    # TODO: implement property tests
    # If hypothesis is available, use @given decorator
    # Otherwise, test with random sizes
    pass

# TODO [EASY]: Test CUDA kernel if available.
#              Skip test gracefully if CUDA not available.
# HINT: import pytest; @pytest.mark.skipif(not torch.cuda.is_available(), ...)

def test_matmul_cuda():
    """Test matmul on CUDA device."""
    if not CUDA_AVAILABLE:
        print("CUDA not available, skipping...")
        return
    
    # TODO: test matmul on CUDA
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What tolerance does float16 need vs float32?
# C2: Which edge case is most likely to reveal kernel bugs?

if __name__ == "__main__":
    print("Testing kernel correctness...\n")
    
    # Test tolerances
    print("Testing tolerance lookup...")
    for dtype in [torch.float32, torch.float16, torch.int8]:
        rtol, atol = get_tolerance_for_dtype(dtype)
        print(f"  {dtype}: rtol={rtol}, atol={atol}")
    
    # Test edge cases
    print("\nTesting edge cases...")
    try:
        test_matmul_edge_cases()
        print("  All edge cases passed ✓")
    except Exception as e:
        print(f"  Edge case failed: {e}")
    
    # Test CUDA
    print("\nTesting CUDA...")
    test_matmul_cuda()
    
    print("\nDone!")
