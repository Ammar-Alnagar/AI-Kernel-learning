"""
Module 01 — Layout Algebra
Exercise 02 — Shape and Stride Algebra

CONCEPT BRIDGE (C++ → DSL):
  C++:  cute::make_layout(cute::make_shape(64, 32, 16), cute::make_stride(1, 64, 64*32))
  DSL:  cute.make_layout((64, 32, 16), stride=(1, 64, 64*32))
  Key:  Column-major 3D layout. Stride tuple can contain expressions.

WHAT YOU'RE BUILDING:
  A 3D column-major layout for a tensor with shape (64, 32, 16). This pattern
  appears in GEMM operand layouts where you want contiguous access along the
  reduction dimension (K). Understanding stride algebra is critical for 
  optimizing memory access patterns in tiled kernels.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create 3D layouts with custom strides
  - Distinguish row-major vs column-major stride patterns
  - Compute linear indices for 3D coordinates
  - Recognize how stride affects memory coalescing

REQUIRED READING (do this before writing any code):
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/layout.html#strides
  - CuTe C++ mental model: stride-1 dimension = contiguous access
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# Answer these questions BEFORE executing the code.
# Write your answers as comments below each question.
# ─────────────────────────────────────────────
# Q1: For a column-major layout with shape (64, 32, 16) and 
#     stride (1, 64, 64*32), what is the linear index of (0, 0, 1)?
# Your answer: 0*1 + 0*64 + 1*2048 = 2048

# Q2: What is the linear index of (1, 0, 0) in this column-major layout?
#     Why is this different from row-major?
# Your answer: 1*1 + 0*64 + 0*2048 = 1. Different because column-major has
#              stride-1 on first dimension, so adjacent d0 coords are contiguous.

# Q3: In a GEMM A-matrix (M x K), which stride pattern gives contiguous
#     access along the K (reduction) dimension: (K, 1) or (1, M)?
# Your answer: (1, M) — column-major. K is the second dimension, but we want
#              contiguous access along K for the reduction loop.

# Q4: For the 3D column-major layout above, what coordinate (d0, d1, d2)
#     maps to linear index 1?
# Your answer: (1, 0, 0) — since d0 has stride-1, index 1 = increment d0 by 1.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# 3D shape: (dim0=64, dim1=32, dim2=16)
D0, D1, D2 = 64, 32, 16

# Column-major stride: stride-1 on the first dimension
# stride = (1, D0, D0*D1)
stride_col_major = (1, D0, D0 * D1)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_stride_algebra(
    results: cute.Tensor,
):
    """
    Compute linear indices for various coordinates in column-major layout.
    
    FILL IN [EASY]: Create the layout and compute indices.
    
    HINT: Column-major means stride-1 on the first (innermost) dimension.
          layout = cute.make_layout((D0, D1, D2), stride=stride_col_major)
    """
    # --- Step 1: Create column-major 3D layout ---
    layout = cute.make_layout((D0, D1, D2), stride=stride_col_major)
    
    # --- Step 2: Index of (0, 0, 1) ---
    results[0] = layout((0, 0, 1))
    
    # --- Step 3: Index of (1, 0, 0) ---
    results[1] = layout((1, 0, 0))
    
    # --- Step 4: Index of (0, 1, 0) ---
    results[2] = layout((0, 1, 0))
    
    # --- Step 5: Find coordinate for linear index 1 ---
    # Use the inverse mapping: given linear index, find coordinate
    flat_layout = cute.cosize(layout)
    coord = flat_layout.inverse(1)
    results[3] = coord[0]  # First dimension of the coordinate
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify stride algebra.
    
    NCU PROFILING COMMAND: Not needed (host-side layout operations).
    """
    
    # Allocate result tensor (5 int32 values)
    result_torch = torch.zeros(5, dtype=torch.int32, device='cuda')
    result_cute = from_dlpack(result_torch)
    
    # Launch kernel
    kernel_stride_algebra(result_cute)
    
    # Copy back
    result_cpu = result_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 01 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  Layout: shape=(64, 32, 16), stride=(1, 64, 2048) [column-major]")
    print(f"\n  Results:")
    print(f"    Index of (0, 0, 1):  {result_cpu[0]}")
    print(f"    Index of (1, 0, 0):  {result_cpu[1]}")
    print(f"    Index of (0, 1, 0):  {result_cpu[2]}")
    print(f"    Coord for index 1:   d0={result_cpu[3]}")
    
    # Verify
    expected_001 = 0 * 1 + 0 * 64 + 1 * 2048  # = 2048
    expected_100 = 1 * 1 + 0 * 64 + 0 * 2048  # = 1
    expected_010 = 0 * 1 + 1 * 64 + 0 * 2048  # = 64
    expected_coord_d0 = 1  # In column-major, index 1 = (1, 0, 0)
    
    print(f"\n  Expected:")
    print(f"    Index of (0, 0, 1):  {expected_001}")
    print(f"    Index of (1, 0, 0):  {expected_100}")
    print(f"    Index of (0, 1, 0):  {expected_010}")
    print(f"    Coord for index 1:   d0={expected_coord_d0}")
    
    passed = (
        result_cpu[0] == expected_001 and
        result_cpu[1] == expected_100 and
        result_cpu[2] == expected_010 and
        result_cpu[3] == expected_coord_d0
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Did your PREDICT answers match the actual results?
# C2: Why does column-major give contiguous access for the first dimension?
# C3: In FlashAttention, the Q matrix is typically (seq_len, heads, dim).
#     Which dimension should have stride-1 for efficient attention computation?
# C4: How would you modify this layout to get row-major behavior?

if __name__ == "__main__":
    run()
