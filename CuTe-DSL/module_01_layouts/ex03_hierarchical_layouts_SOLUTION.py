"""
Module 01 — Layout Algebra
Exercise 03 — Hierarchical Layouts

CONCEPT BRIDGE (C++ → DSL):
  C++:  cute::make_layout(make_shape(M, N), make_stride(N, 1))
        where M = (m1, m2) and N = (n1, n2) are nested shapes
  DSL:  cute.make_layout(((m1, m2), (n1, n2)), stride=((s1, s2), (t1, t2)))
  Key:  Hierarchical layouts compose multiple levels of tiling.

WHAT YOU'RE BUILDING:
  A hierarchical layout representing a 2D tile structure: ((4, 32), (2, 16)).
  This models a (128, 32) matrix viewed as a 2×2 grid of tiles, where each
  tile is (64, 16). Hierarchical layouts are essential for:
  - Multi-level tiling in GEMM (CTA-level → warp-level → thread-level)
  - GQA (Grouped Query Attention) where KV heads are tiled separately
  - FlashAttention's block-tiling over Q and K/V sequences

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create hierarchical layouts with nested tuple shapes
  - Understand how hierarchical coordinates map to linear indices
  - Use hierarchical layouts for multi-level tiling

REQUIRED READING (do this before writing any code):
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/layout.html#hierarchical-layouts
  - CuTe C++ mental model: hierarchical = nested shape/stride tuples
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
# Q1: For a hierarchical layout with shape ((4, 32), (2, 16)),
#     what is the total number of elements?
# Your answer: 4 * 32 * 2 * 16 = 4096

# Q2: If we use row-major stride at each level, what would the stride
#     tuple look like? Hint: think about flattening each level.
# Your answer: ((32*2*16, 2*16), (16, 1)) = ((1024, 32), (16, 1))
#              Outer M stride: 1024 (size of one M-slice)
#              Inner M stride: 32 (size of N dimension)
#              Outer N stride: 16 (size of inner N tile)
#              Inner N stride: 1 (contiguous)

# Q3: In FlashAttention-2, the sequence is tiled into blocks of 64 or 128.
#     How would you represent a (seq_len=4096, block_size=64) tiling as a
#     hierarchical shape?
# Your answer: ((64, 64), ...) where 64*64=4096, or more simply:
#              (64, 64) for 64 blocks of size 64

# Q4: What is the difference between shape ((4, 32), (2, 16)) and 
#     shape (4, 32, 2, 16) in terms of coordinate access?
# Your answer: Hierarchical ((4,32),(2,16)) groups dimensions semantically:
#              coord = ((m_outer, m_inner), (n_outer, n_inner))
#              Flat (4,32,2,16) treats all 4 dims equally:
#              coord = (d0, d1, d2, d3)
#              Hierarchical enables tile-based reasoning.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Hierarchical shape: ((m_outer, m_inner), (n_outer, n_inner))
# This represents a (4*32=128, 2*16=32) matrix viewed as 2D tiles
hierarchical_shape = ((4, 32), (2, 16))

# For row-major at each level:
# - Inner N level: stride (16, 1) for the (2, 16) tile
# - Outer M level: stride (32*2*16, ...) for stepping between M tiles
# We'll compute this programmatically below.


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_hierarchical_layout(
    results: cute.Tensor,
):
    """
    Create and query a hierarchical layout.
    
    FILL IN [MEDIUM]: Create hierarchical layout and compute indices.
    
    HINT: Hierarchical layouts use nested tuples. The coordinate is also nested:
          layout(((m_outer, m_inner), (n_outer, n_inner)))
          
    For row-major at each level, think about:
    - Within a tile: row-major stride
    - Between tiles: stride = tile_size * num_tiles_after
    """
    # --- Step 1: Create hierarchical layout ---
    # stride = ((32*2*16, 2*16), (16, 1))
    #        = ((1024, 32), (16, 1))
    layout = cute.make_layout(hierarchical_shape, stride=((1024, 32), (16, 1)))
    
    # --- Step 2: Query layout properties ---
    # Store the rank (number of dimensions at top level)
    results[0] = cute.rank(layout)
    
    # Store the depth (total number of leaf dimensions)
    results[1] = cute.depth(layout)
    
    # Store the total size (product of all shape elements)
    flat = cute.cosize(layout)
    results[2] = flat.size()
    
    # --- Step 3: Access with hierarchical coordinate ---
    # Coordinate ((1, 8), (0, 4)) means:
    #   m_outer=1, m_inner=8 → row 1*32+8 = 40
    #   n_outer=0, n_inner=4 → col 0*16+4 = 4
    # Linear index = 40 * 32 + 4 = 1284
    idx = layout(((1, 8), (0, 4)))
    results[3] = idx
    
    # --- Step 4: Access with flat coordinate (after cosize) ---
    # The same element should be accessible via flat coordinate
    flat_layout = cute.cosize(layout)
    # Verify: flat_layout(1284) should give us back 1284 (identity on flat)
    results[4] = flat_layout(1284)
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify hierarchical layout properties.
    
    NCU PROFILING COMMAND: Not needed (host-side layout operations).
    """
    
    # Allocate result tensor (5 int32 values)
    result_torch = torch.zeros(5, dtype=torch.int32, device='cuda')
    result_cute = from_dlpack(result_torch)
    
    # Launch kernel
    kernel_hierarchical_layout(result_cute)
    
    # Copy back
    result_cpu = result_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 01 — Exercise 03 Results")
    print("=" * 60)
    print(f"\n  Hierarchical layout: shape={hierarchical_shape}")
    print(f"  Interpreted as: (4*32=128, 2*16=32) matrix with 2D tiling")
    print(f"\n  Results:")
    print(f"    Rank (top-level dims):    {result_cpu[0]}")
    print(f"    Depth (leaf dims):        {result_cpu[1]}")
    print(f"    Total size (elements):    {result_cpu[2]}")
    print(f"    Index of ((1,8),(0,4)):   {result_cpu[3]}")
    print(f"    Flat layout(1284):        {result_cpu[4]}")
    
    # Verify
    expected_rank = 2  # Two top-level dimensions: M and N
    expected_depth = 4  # Four leaf dimensions: m_outer, m_inner, n_outer, n_inner
    expected_size = 4 * 32 * 2 * 16  # = 4096
    # ((1, 8), (0, 4)) → row = 1*32+8 = 40, col = 0*16+4 = 4
    # Row-major index = 40 * 32 + 4 = 1284
    expected_idx = 1284
    expected_flat = 1284  # Identity on flat layout
    
    print(f"\n  Expected:")
    print(f"    Rank (top-level dims):    {expected_rank}")
    print(f"    Depth (leaf dims):        {expected_depth}")
    print(f"    Total size (elements):    {expected_size}")
    print(f"    Index of ((1,8),(0,4)):   {expected_idx}")
    print(f"    Flat layout(1284):        {expected_flat}")
    
    passed = (
        result_cpu[0] == expected_rank and
        result_cpu[1] == expected_depth and
        result_cpu[2] == expected_size and
        result_cpu[3] == expected_idx and
        result_cpu[4] == expected_flat
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Did your PREDICT answers match the actual results?
# C2: How does hierarchical layout help with multi-level tiling in GEMM?
# C3: In FlashAttention, you tile both Q (seq_len) and K/V (seq_len).
#     How would you represent a hierarchical layout for both dimensions?
# C4: What's the advantage of hierarchical vs flat layout for tile-based kernels?

if __name__ == "__main__":
    run()
