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
  - CUTLASS Python DSL layout docs: https://nvidia.github.io/cutlass-dsl/
  - This curriculum's API map: ../API_MIGRATION_4x.md

API NOTE (4.8):
  Layouts build MLIR, so `cute.make_layout` may only be called from inside a
  `@cute.kernel` or `@cute.jit` function — never from plain Python.
  A `@cute.jit` host function cannot write to a tensor, so the store below
  happens in the kernel, guarded by `tidx == 0`.
"""

import pathlib
import sys

import torch

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
import cutedsl_utils as utils  # noqa: E402

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
# Shape and stride for a row-major (128, 64) matrix.
# Module-level Python ints are captured as compile-time constants by the DSL.
M, N = 128, 64


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cute.kernel
def kernel_layout_query(layout_result: cute.Tensor):
    """
    Query layout properties and store results in a tensor.

    SOLUTION:
      `cute.crd2idx(coord, layout)` is the 4.x spelling of `layout(coord)` in C++.
      The argument order is (coord, layout) — the reverse of the C++ member call.
    """
    # --- Step 1: Create the layout ---
    layout = cute.make_layout((M, N), stride=(N, 1))

    # --- Step 2: Compute linear index for (5, 10) ---
    # --- Step 3: Compute linear index for (32, 0) ---
    # --- Step 4: Get total size (cosize) ---
    tidx, _, _ = cute.arch.thread_idx()
    if tidx == 0:
        layout_result[0] = cute.crd2idx((5, 10), layout)
        layout_result[1] = cute.crd2idx((32, 0), layout)
        layout_result[2] = cute.cosize(layout)


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
@cute.jit
def launch(layout_result: cute.Tensor):
    """
    Launch configuration lives in the host function, not in the kernel.

    grid/block replace the old `kernel(args, grid=..., block=...)` form.
    """
    kernel_layout_query(layout_result).launch(grid=[1, 1, 1], block=[32, 1, 1])


def run():
    """
    Run the kernel and verify layout mapping.

    NCU PROFILING COMMAND (copy-paste to terminal):
    # Not needed for this exercise — layout is a compile-time data structure.
    # Profiling starts in Module 03 (TiledCopy).
    """
    # Allocate result tensor (3 int32 values)
    result_torch = torch.zeros(3, dtype=torch.int32, device="cuda")

    # Launch kernel
    launch(from_dlpack(result_torch))
    torch.cuda.synchronize()

    got = result_torch.cpu().tolist()

    utils.banner("Module 01 — Exercise 01 Results")
    print(f"  Layout: shape=({M}, {N}), stride=({N}, 1) [row-major]\n")
    print("  results:")
    print(f"    linear index of (5, 10): {got[0]}")
    print(f"    linear index of (32, 0): {got[1]}")
    print(f"    cosize:                  {got[2]}")
    print()

    # Verify correctness
    expected = (5 * N + 10, 32 * N, M * N)
    print("  expected:")
    print(f"    linear index of (5, 10): {expected[0]}")
    print(f"    linear index of (32, 0): {expected[1]}")
    print(f"    cosize:                  {expected[2]}")
    print()

    passed = True
    passed &= utils.check("index(5,10)", got[0], expected[0])
    passed &= utils.check("index(32,0)", got[1], expected[1])
    passed &= utils.check("cosize", got[2], expected[2])
    utils.rule()
    print(f"  {'PASSED' if passed else 'FAILED'}")
    utils.rule()
    print()
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