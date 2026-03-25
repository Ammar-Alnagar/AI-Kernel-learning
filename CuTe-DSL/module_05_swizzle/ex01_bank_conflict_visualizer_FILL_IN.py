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
# Your answer:

# Q2: For a row-major (32, 32) matrix, if 32 threads read column 0,
#     how many-way bank conflict occurs?
# Your answer:

# Q3: What access pattern causes no bank conflicts for 32 threads?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# SMEM matrix dimensions
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
    
    FILL IN [MEDIUM]: Calculate bank indices and count conflicts.
    
    HINT: Bank index = (linear_address) % NUM_BANKS
          For row-major: linear_address = row * N + col
    """
    # --- Pattern 1: Column-wise access (worst case) ---
    # All threads read column 0 from different rows
    # TODO: For each row in 0..31, calculate bank = (row * N + 0) % 32
    #       Count how many accesses hit each bank
    
    # --- Pattern 2: Row-wise access (best case) ---
    # All threads read row 0 from different columns
    # TODO: For each col in 0..31, calculate bank = (0 * N + col) % 32
    
    # --- Pattern 3: Diagonal access ---
    # Thread i reads element (i, i)
    # TODO: For each i in 0..31, calculate bank = (i * N + i) % 32
    
    # Store max conflict count for each pattern in results[0:3]
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and analyze bank conflict patterns.
    """
    
    # Bank count tensor (32 banks)
    bank_counts_torch = torch.zeros(NUM_BANKS, dtype=torch.int32, device='cuda')
    bank_counts_cute = from_dlpack(bank_counts_torch)
    
    # Results: max conflicts for each pattern
    results_torch = torch.zeros(6, dtype=torch.int32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch
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
    
    # Verify expected patterns
    # Column-wise: all rows map to same bank → 32-way conflict
    # Row-wise: each col maps to different bank → no conflict
    # Diagonal: depends on N
    
    passed = (
        results_cpu[0] == 32 and  # Column-wise = 32-way
        results_cpu[1] == 1      # Row-wise = no conflict
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
