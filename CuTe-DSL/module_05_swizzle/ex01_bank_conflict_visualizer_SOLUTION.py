"""
Module 05 — Swizzle
Exercise 01 — Bank Conflict Visualizer

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Analyze bank conflicts for a given access pattern
        for (int col = 0; col < N; ++col) {
            int bank = (row * N + col) % 32;  // 32 banks
        }
  DSL:  # Same analysis in Python
        bank = (row * N + col) % 32
  Key:  SMEM has 32 banks. Conflicts occur when multiple threads access same bank.

WHAT YOU'RE BUILDING:
  A bank conflict visualizer that shows how different access patterns cause
  bank conflicts. This tool helps understand why swizzling is necessary for
  high-performance SMEM access.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Calculate bank indices for SMEM accesses
  - Identify bank conflict patterns
  - Understand how swizzling distributes accesses

REQUIRED READING:
  - CUDA PTX docs: https://docs.nvidia.com/cuda/parallel-thread-execution/#shared-memory
  - NVIDIA blog: https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: How many memory banks does modern NVIDIA SMEM have?
# Your answer: 32 banks

# Q2: For a row-major (32, 32) matrix, if 32 threads read column 0,
#     how many-way bank conflict occurs?
# Your answer: 32-way conflict (all rows map to same bank)

# Q3: What access pattern causes no bank conflicts for 32 threads?
# Your answer: Row-wise access (consecutive columns)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
M, N = 32, 32
NUM_BANKS = 32


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_bank_conflict_visualizer(
    bank_counts: cute.Tensor,
    results: cute.Tensor,
):
    """
    Visualize bank conflicts for different access patterns.
    """
    # Clear bank counts
    for i in range(NUM_BANKS):
        bank_counts[i] = 0
    
    # --- Pattern 1: Column-wise access (worst case) ---
    # All threads read column 0 from different rows
    for row in range(32):
        addr = row * N + 0
        bank = addr % NUM_BANKS
        bank_counts[bank] += 1
    
    # Find max conflict
    max_col = 0
    for i in range(NUM_BANKS):
        if bank_counts[i] > max_col:
            max_col = bank_counts[i]
    results[0] = max_col
    
    # Clear for next pattern
    for i in range(NUM_BANKS):
        bank_counts[i] = 0
    
    # --- Pattern 2: Row-wise access (best case) ---
    for col in range(32):
        addr = 0 * N + col
        bank = addr % NUM_BANKS
        bank_counts[bank] += 1
    
    max_row = 0
    for i in range(NUM_BANKS):
        if bank_counts[i] > max_row:
            max_row = bank_counts[i]
    results[1] = max_row
    
    # Clear for next pattern
    for i in range(NUM_BANKS):
        bank_counts[i] = 0
    
    # --- Pattern 3: Diagonal access ---
    for i in range(32):
        addr = i * N + i
        bank = addr % NUM_BANKS
        bank_counts[bank] += 1
    
    max_diag = 0
    for i in range(NUM_BANKS):
        if bank_counts[i] > max_diag:
            max_diag = bank_counts[i]
    results[2] = max_diag
    
    # Store expected values
    results[3] = 32  # Expected column-wise
    results[4] = 1   # Expected row-wise
    results[5] = max_diag
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and analyze bank conflict patterns.
    """
    
    bank_counts_torch = torch.zeros(NUM_BANKS, dtype=torch.int32, device='cuda')
    bank_counts_cute = from_dlpack(bank_counts_torch)
    
    results_torch = torch.zeros(6, dtype=torch.int32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_bank_conflict_visualizer[1, 1](bank_counts_cute, results_cute)
    
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 05 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  SMEM Configuration: {NUM_BANKS} banks")
    print(f"  Matrix: ({M}, {N})")
    print(f"\n  Bank Conflict Analysis:")
    print(f"    Column-wise access (worst):  {results_cpu[0]}-way conflict")
    print(f"    Row-wise access (best):      {results_cpu[1]}-way conflict")
    print(f"    Diagonal access:             {results_cpu[2]}-way conflict")
    
    passed = (
        results_cpu[0] == 32 and
        results_cpu[1] == 1
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: Why does column-wise access cause 32-way bank conflicts?
# C2: How does swizzling help reduce bank conflicts?
# C3: In GEMM, which access pattern (row/column) is more common?
# C4: What is the performance impact of 32-way bank conflicts?

if __name__ == "__main__":
    run()
