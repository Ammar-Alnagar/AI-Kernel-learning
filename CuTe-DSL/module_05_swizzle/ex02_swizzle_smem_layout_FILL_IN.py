"""
Module 05 — Swizzle
Exercise 02 — Swizzle SMEM Layout

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto swizzled_layout = composition(Swizzle<6,3,3>{}, row_major_layout);
  DSL:  swizzled_layout = cute.composition(cute.Swizzle(6, 3, 3), row_major_layout)
  Key:  Swizzling XORs address bits to distribute accesses across banks.

WHAT YOU'RE BUILDING:
  A swizzled SMEM layout that eliminates bank conflicts for common GEMM access
  patterns. The Swizzle(6, 3, 3) parameters XOR bits 3-5 into bits 6-8, spreading
  consecutive addresses across different banks.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create swizzled layouts using composition
  - Understand swizzle parameters (B, M, S)
  - Verify bank conflict reduction

REQUIRED READING:
  - CUTLASS swizzle docs: https://nvidia.github.io/cutlass-dsl/cute/swizzle.html
  - CuTe C++ mental model: Swizzle is a layout transformation
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does Swizzle(6, 3, 3) do to the address bits?
# Your answer:

# Q2: For a 128-byte SMEM line (32 × 4-byte elements), which swizzle
#     parameters would you use?
# Your answer:

# Q3: Does swizzling change the logical shape of the tensor?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# SMEM matrix dimensions (must be power of 2 for swizzling)
M, N = 32, 32

# Swizzle parameters for 128-byte lines
# B=6, M=3, S=3 is common for GEMM
SWIZZLE_B, SWIZZLE_M, SWIZZLE_S = 6, 3, 3


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_swizzle_smem_layout(
    smem_ptr: cute.Pointer,
    results: cute.Tensor,
):
    """
    Create and verify a swizzled SMEM layout.
    
    FILL IN [HARD]: Apply swizzle to SMEM layout and verify bank distribution.
    
    HINT: swizzle = cute.Swizzle(SWIZZLE_B, SWIZZLE_M, SWIZZLE_S)
          row_major = cute.make_layout((M, N), stride=(N, 1))
          swizzled = cute.composition(swizzle, row_major)
    """
    # --- Step 1: Create row-major layout ---
    # TODO: row_major = cute.make_layout((M, N), stride=(N, 1))
    
    # --- Step 2: Create swizzle ---
    # TODO: swizzle = cute.Swizzle(SWIZZLE_B, SWIZZLE_M, SWIZZLE_S)
    
    # --- Step 3: Compose to get swizzled layout ---
    # TODO: swizzled_layout = cute.composition(swizzle, row_major)
    
    # --- Step 4: Create SMEM tensor with swizzled layout ---
    # TODO: smem_tensor = cute.make_smem_tensor(smem_ptr, swizzled_layout)
    
    # --- Step 5: Verify swizzle by checking address mapping ---
    # Compare linear index of (0, 1) vs (0, 0) with and without swizzle
    # Store in results[0:4]
    
    # --- Step 6: Check bank distribution ---
    # For consecutive addresses, count unique banks accessed
    # Store in results[4:6]
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify swizzle layout.
    
    NCU PROFILING COMMAND:
    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
        --set full --target-processes all \
        python ex02_swizzle_smem_layout_FILL_IN.py
    """
    
    # Allocate SMEM
    smem_torch = torch.zeros(M * N, dtype=torch.float32, device='cuda')
    smem_ptr = from_dlpack(smem_torch)
    
    # Results
    results_torch = torch.zeros(8, dtype=torch.int32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch
    kernel_swizzle_smem_layout[1, 1](smem_ptr, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 05 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  Swizzle Configuration:")
    print(f"    Parameters: Swizzle({SWIZZLE_B}, {SWIZZLE_M}, {SWIZZLE_S})")
    print(f"    Matrix: ({M}, {N})")
    print(f"\n  Layout Comparison:")
    print(f"    Row-major (0,0): {results_cpu[0]}")
    print(f"    Swizzled (0,0):  {results_cpu[1]}")
    print(f"    Row-major (0,1): {results_cpu[2]}")
    print(f"    Swizzled (0,1):  {results_cpu[3]}")
    print(f"\n  Bank Distribution:")
    print(f"    Without swizzle: {results_cpu[4]} unique banks")
    print(f"    With swizzle:    {results_cpu[5]} unique banks")
    
    # Swizzle should preserve (0,0) mapping but change (0,1)
    passed = (
        results_cpu[0] == results_cpu[1] and  # (0,0) unchanged
        results_cpu[2] != results_cpu[3] and  # (0,1) changed by swizzle
        results_cpu[5] > results_cpu[4]       # More banks with swizzle
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does swizzling change the address mapping?
# C2: Why does Swizzle(6, 3, 3) work well for 128-byte lines?
# C3: In FlashAttention, where would you apply swizzled layouts?
# C4: Does swizzling affect the logical (row, col) → value semantics?

if __name__ == "__main__":
    run()
