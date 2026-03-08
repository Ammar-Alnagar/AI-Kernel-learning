"""
Module 02 — Tensors
Exercise 04 — Slicing and Views

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto view = tensor[make_coord(_, 0)];  // First column
  DSL:  view = tensor[:, 0]  # Python slice syntax
  Key:  Python slice syntax creates zero-copy views with updated layouts.

WHAT YOU'RE BUILDING:
  Tensor views using Python's native slice syntax. Views are zero-copy — they
  reuse the underlying memory with a modified layout. This is critical for
  efficient tensor manipulation in attention kernels where you frequently
  extract rows, columns, or tiles without copying.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create tensor views using Python slice syntax
  - Understand that views are zero-copy (layout transformation only)
  - Use views for efficient row/column/tile extraction

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tensor.html#views-and-slicing
  - NumPy slicing reference (similar semantics): https://numpy.org/doc/stable/reference/arrays.indexing.html
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: If we have a tensor with shape (64, 32) and create a view tensor[:, 0],
#     what is the shape of the view?
# Your answer: (64,) — a 1D view of the first column

# Q2: Does creating a view copy any data?
# Your answer: No — views are zero-copy. Only the layout changes.

# Q3: If the original tensor has row-major stride (32, 1), what is the
#     stride of the view tensor[10:20, :] (rows 10-19)?
# Your answer: Still (32, 1) — the stride is preserved, just the shape changes


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
M, N = 64, 32

# Create a tensor with known pattern: element (i, j) = i * N + j
torch_tensor = torch.arange(M * N, dtype=torch.float32, device='cuda').reshape(M, N)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_slicing_views(
    tensor: cute.Tensor,
    results: cute.Tensor,
):
    """
    Demonstrate tensor slicing and views.
    
    FILL IN [MEDIUM]: Create views using slice syntax and verify they're zero-copy.
    
    HINT: Use Python slice syntax: tensor[start:end], tensor[:, col], tensor[row, :]
          Views share the same underlying memory as the original.
    """
    # --- Step 1: Extract first column ---
    col0 = tensor[:, 0]
    results[0] = col0[10]  # Should be element (10, 0) = 320
    
    # --- Step 2: Extract row 5 ---
    row5 = tensor[5, :]
    results[1] = row5[10]  # Should be element (5, 10) = 170
    
    # --- Step 3: Extract a tile (rows 10-19, cols 5-14) ---
    tile = tensor[10:20, 5:15]
    results[2] = tile[0, 0]  # Should be element (10, 5) = 325
    results[3] = tile[5, 5]  # Should be element (15, 10) = 490
    
    # --- Step 4: Verify zero-copy by modifying through view ---
    col0[0] = 999.0
    results[4] = tensor[0, 0]  # Should now be 999.0
    
    # --- Step 5: Store view shapes ---
    results[5] = float(col0.shape[0])  # = 64
    results[6] = float(row5.shape[0])  # = 32
    results[7] = float(tile.shape[0] * tile.shape[1])  # = 10 * 10 = 100
    results[8] = float(tensor.shape[0] * tensor.shape[1])  # = 2048
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify slicing/view behavior.
    
    NCU PROFILING COMMAND: Not needed (host-side view operations).
    """
    
    # Convert to CuTe tensor
    tensor = from_dlpack(torch_tensor)
    
    # Allocate results tensor
    results_torch = torch.zeros(9, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel
    kernel_slicing_views[1, 1](tensor, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 02 — Exercise 04 Results")
    print("=" * 60)
    print(f"\n  Original Tensor: shape=({M}, {N})")
    print(f"\n  Results:")
    print(f"    tensor[:, 0][10]:     {results_cpu[0]}")
    print(f"    tensor[5, :][10]:     {results_cpu[1]}")
    print(f"    tensor[10:20,5:15][0,0]: {results_cpu[2]}")
    print(f"    tensor[10:20,5:15][5,5]: {results_cpu[3]}")
    print(f"    After view write tensor[0,0]: {results_cpu[4]}")
    print(f"    col0 view size:       {results_cpu[5]}")
    print(f"    row5 view size:       {results_cpu[6]}")
    print(f"    tile view size:       {results_cpu[7]}")
    print(f"    Original tensor size: {results_cpu[8]}")
    
    # Verify
    expected_col0_10 = 10 * 32 + 0  # = 320
    expected_row5_10 = 5 * 32 + 10  # = 170
    expected_tile_00 = 10 * 32 + 5  # = 325
    expected_tile_55 = 15 * 32 + 10  # = 490
    expected_modified = 999.0  # After view write
    
    passed = (
        results_cpu[0] == expected_col0_10 and
        results_cpu[1] == expected_row5_10 and
        results_cpu[2] == expected_tile_00 and
        results_cpu[3] == expected_tile_55 and
        results_cpu[4] == expected_modified
    )
    
    print(f"\n  Expected:")
    print(f"    tensor[:, 0][10]:     {expected_col0_10}")
    print(f"    tensor[5, :][10]:     {expected_row5_10}")
    print(f"    tensor[10:20,5:15][0,0]: {expected_tile_00}")
    print(f"    tensor[10:20,5:15][5,5]: {expected_tile_55}")
    print(f"    After view write:     {expected_modified}")
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: Are views zero-copy? How can you verify this?
# C2: In FlashAttention, where would you use views to extract Q, K, V slices?
# C3: What's the difference between tensor[5, :] and tensor[5:6, :]?
# C4: How does CuTe's view semantics compare to NumPy's?

if __name__ == "__main__":
    run()
