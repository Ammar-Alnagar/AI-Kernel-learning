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
# Your answer: XORs bits [3:6) into bits [6:9). Specifically:
#              addr[8:6] ^= addr[5:3]

# Q2: For a 128-byte SMEM line (32 × 4-byte elements), which swizzle
#     parameters would you use?
# Your answer: Swizzle(6, 3, 3) - standard for 128-byte lines

# Q3: Does swizzling change the logical shape of the tensor?
# Your answer: No - logical (row, col) access is unchanged.
#              Only the physical address mapping changes.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
M, N = 32, 32
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
    """
    # --- Step 1: Create row-major layout ---
    row_major = cute.make_layout((M, N), stride=(N, 1))
    
    # --- Step 2: Create swizzle ---
    swizzle = cute.Swizzle(SWIZZLE_B, SWIZZLE_M, SWIZZLE_S)
    
    # --- Step 3: Compose to get swizzled layout ---
    swizzled_layout = cute.composition(swizzle, row_major)
    
    # --- Step 4: Create SMEM tensor with swizzled layout ---
    smem_tensor = cute.make_smem_tensor(smem_ptr, swizzled_layout)
    
    # --- Step 5: Verify swizzle by checking address mapping ---
    # Row-major (0, 0) = 0, Swizzled (0, 0) should also be 0
    results[0] = row_major((0, 0))
    results[1] = swizzled_layout((0, 0))
    
    # Row-major (0, 1) = 1, Swizzled (0, 1) should be different
    results[2] = row_major((0, 1))
    results[3] = swizzled_layout((0, 1))
    
    # --- Step 6: Check bank distribution ---
    # Count unique banks for consecutive addresses
    banks_no_swizzle = 0
    banks_with_swizzle = 0
    
    seen_no_swizzle = [0] * 32
    seen_with_swizzle = [0] * 32
    
    for i in range(32):
        addr_normal = i
        addr_swizzled = swizzled_layout((0, i))
        
        bank_normal = addr_normal % 32
        bank_swizzled = addr_swizzled % 32
        
        seen_no_swizzle[bank_normal] = 1
        seen_with_swizzle[bank_swizzled] = 1
    
    for i in range(32):
        banks_no_swizzle += seen_no_swizzle[i]
        banks_with_swizzle += seen_with_swizzle[i]
    
    results[4] = banks_no_swizzle
    results[5] = banks_with_swizzle
    
    # Store expected values
    results[6] = 0  # (0,0) should map to 0
    results[7] = 32  # With swizzle, all 32 banks should be used
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify swizzle layout.
    """
    
    smem_torch = torch.zeros(M * N, dtype=torch.float32, device='cuda')
    smem_ptr = from_dlpack(smem_torch)
    
    results_torch = torch.zeros(8, dtype=torch.int32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
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
    
    passed = (
        results_cpu[0] == results_cpu[1] and
        results_cpu[2] != results_cpu[3] and
        results_cpu[5] >= results_cpu[4]
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
