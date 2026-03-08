"""
Module 05 — Type Hints
Exercise 02 — Tensor Types

WHAT YOU'RE BUILDING:
  Tensor type hints document expected shapes and dtypes for kernel inputs.
  While Python can't enforce shape constraints at runtime, type hints
  document intent and enable IDE autocomplete for tensor methods.

OBJECTIVE:
  - Use torch.Tensor in type hints
  - Document expected shapes in comments/conventions
  - Understand TypeAlias for custom tensor types
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: How do you hint a 2D tensor vs a 1D tensor in Python?
# Q2: What's a TypeAlias and why use it for tensor shapes?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import Tuple, TypeAlias
import torch

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Create type aliases for common tensor shapes.
#              This documents expected shapes at a glance.
# HINT: Matrix2D: TypeAlias = torch.Tensor  # (M, N)

# Type alias for a 2D matrix (M, N)
Matrix2D: TypeAlias = torch.Tensor

# TODO: Create type aliases for:
#       - Vector1D: 1D tensor (N,)
#       - Tensor3D: 3D tensor (B, M, N)
#       - WeightTensor: 2D tensor for weights (out_features, in_features)

Vector1D: TypeAlias = None  # TODO: replace
Tensor3D: TypeAlias = None  # TODO: replace
WeightTensor: TypeAlias = None  # TODO: replace

# TODO [EASY]: Annotate this matmul function with tensor type hints.
#              Document expected shapes in comments.
# HINT: def matmul(a: Matrix2D, b: Matrix2D) -> Matrix2D:

def matmul(a, b):
    """Matrix multiplication.
    
    Args:
        a: (M, K) tensor
        b: (K, N) tensor
    
    Returns:
        (M, N) tensor
    """
    # TODO: add type hints to signature
    return torch.matmul(a, b)

# TODO [MEDIUM]: Annotate this batched matmul function.
#              Use Tensor3D for batched inputs.
# HINT: def batch_matmul(a: Tensor3D, b: Tensor3D) -> Tensor3D:

def batch_matmul(a, b):
    """Batched matrix multiplication.
    
    Args:
        a: (B, M, K) tensor
        b: (B, K, N) tensor
    
    Returns:
        (B, M, N) tensor
    """
    # TODO: add type hints
    return torch.bmm(a, b)

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What's the benefit of TypeAlias for tensor shapes?
# C2: Can Python enforce shape constraints at runtime from type hints?

if __name__ == "__main__":
    print("Testing tensor type hints...")
    
    # Create test tensors
    a = torch.randn(10, 20)
    b = torch.randn(20, 30)
    
    result = matmul(a, b)
    print(f"matmul result shape: {result.shape}")
    
    # Batched matmul
    a_batch = torch.randn(4, 10, 20)
    b_batch = torch.randn(4, 20, 30)
    
    result_batch = batch_matmul(a_batch, b_batch)
    print(f"batch_matmul result shape: {result_batch.shape}")
    
    print("\nDone!")
