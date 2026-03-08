"""
Module 01 — Layout Algebra
Exercise 01 — make_layout Basics

CONCEPT BRIDGE (C++ → DSL):
  C++:  cute::make_layout(cute::make_shape(128, 64), cute::make_stride(64, 1))
  DSL:  cute.make_layout((128, 64), stride=(64, 1))
  Key:  Python tuples replace make_shape/make_stride. Same algebra underneath.

WHAT YOU'RE BUILDING:
  A simple row-major 2D layout representing a (128, 64) matrix. This is the
  foundational building block for every tiled kernel — GEMM, attention, conv.
  You'll verify the layout mapping by computing indices manually.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create 2D layouts using Python tuple syntax
  - Compute the linear index for given (row, col) coordinates
  - Verify layout mapping matches expected row-major behavior

REQUIRED READING (do this before writing any code):
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/layout.html
  - Example to study: cutlass/examples/python/CuTeDSL/layout_basics.py
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
# Q1: For a row-major layout with shape (128, 64) and stride (64, 1),
#     what is the linear index of coordinate (row=5, col=10)?
# Your answer: 5 * 64 + 10 = 330

# Q2: What is the total size (number of elements) of this layout?
# Your answer: 128 * 64 = 8192

# Q3: If we access coordinate (32, 0), what linear index do we get?
#     How does this confirm row-major ordering?
# Your answer: 32 * 64 + 0 = 2048. Confirms row-major because each row
#              is contiguous, and we skip 32 full rows of 64 elements.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Define shape and stride for a row-major (128, 64) matrix
M, N = 128, 64


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_layout_query(
    layout_result: cute.Tensor,
):
    """
    Query layout properties and store results in a tensor.
    
    FILL IN [EASY]: Create the layout and compute indices for test coordinates.
    
    HINT: Use cute.make_layout(shape_tuple, stride=stride_tuple)
    """
    # --- Step 1: Create the layout ---
    layout = cute.make_layout((M, N), stride=(N, 1))
    
    # --- Step 2: Compute linear index for (5, 10) ---
    layout_result[0] = layout((5, 10))
    
    # --- Step 3: Compute linear index for (32, 0) ---
    layout_result[1] = layout((32, 0))
    
    # --- Step 4: Get total size (cosize) ---
    flat_layout = cute.cosize(layout)
    layout_result[2] = flat_layout.size()
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify layout mapping.
    
    NCU PROFILING COMMAND (copy-paste to terminal):
    # Not needed for this exercise — layout is host-side data structure.
    # Profiling starts in Module 03 (TiledCopy).
    
    METRICS TO FOCUS ON (for later modules):
    - l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum: SMEM bank conflicts
    - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum: Global load throughput
    """
    
    # Allocate result tensor (3 int32 values)
    result_torch = torch.zeros(3, dtype=torch.int32, device='cuda')
    result_cute = from_dlpack(result_torch)
    
    # Launch kernel
    kernel_layout_query(result_cute)
    
    # Copy back and verify
    result_cpu = result_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 01 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  Layout: shape=(128, 64), stride=(64, 1) [row-major]")
    print(f"\n  Results:")
    print(f"    Linear index of (5, 10):  {result_cpu[0]}")
    print(f"    Linear index of (32, 0):  {result_cpu[1]}")
    print(f"    Total size (cosize):      {result_cpu[2]}")
    
    # Verify correctness
    expected_idx_5_10 = 5 * 64 + 10  # row-major: row * stride_row + col
    expected_idx_32_0 = 32 * 64 + 0
    expected_size = 128 * 64
    
    print(f"\n  Expected:")
    print(f"    Linear index of (5, 10):  {expected_idx_5_10}")
    print(f"    Linear index of (32, 0):  {expected_idx_32_0}")
    print(f"    Total size:               {expected_size}")
    
    passed = (
        result_cpu[0] == expected_idx_5_10 and
        result_cpu[1] == expected_idx_32_0 and
        result_cpu[2] == expected_size
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Did your PREDICT answers match? What was different and why?
# C2: How does the tuple syntax (M, N) compare to make_shape(M, N) 
#     in terms of readability?
# C3: In a production GEMM kernel, where would you use this row-major 
#     layout? (Hint: think about the C/output matrix)
# C4: What would change if we used column-major stride (1, M) instead?

if __name__ == "__main__":
    run()
