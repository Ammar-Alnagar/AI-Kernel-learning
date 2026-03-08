"""
Module 02 — Tensors
Exercise 02 — SMEM Tensor with Shared Memory Pointer

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto smem_tensor = make_tensor(make_smem_ptr(smem_ptr), layout);
  DSL:  smem_tensor = cute.make_smem_tensor(smem_ptr, layout)
  Key:  SMEM tensors use shared memory pointers for on-chip, low-latency access.

WHAT YOU'RE BUILDING:
  A shared memory tensor for buffering data in the on-chip SMEM. This is the
  critical intermediate storage in GMEM→SMEM→RMEM pipelines used in every
  production GEMM and attention kernel. You'll allocate SMEM, create the tensor,
  and verify bank-conflict-free access patterns.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Allocate shared memory and create SMEM tensors
  - Understand SMEM's role in the memory hierarchy
  - Recognize bank conflict patterns in SMEM access

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tensor.html#shared-memory
  - NVIDIA PTX docs: https://docs.nvidia.com/cuda/parallel-thread-execution/#shared-memory
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What is the typical size of shared memory per SM on Ampere (SM80)?
# Your answer:

# Q2: Why do we need to explicitly allocate SMEM vs GMEM which is allocated by PyTorch?
# Your answer:

# Q3: For a (64, 32) matrix in SMEM with row-major layout, which access pattern
#     causes bank conflicts: reading along rows or reading along columns?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
M, N = 64, 32
smem_size_bytes = M * N * 4  # float32 = 4 bytes


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_smem_tensor(
    smem_ptr: cute.Pointer,
    results: cute.Tensor,
):
    """
    Create SMEM tensor and demonstrate shared memory access.
    
    FILL IN [EASY]: Create SMEM tensor from pointer and perform accesses.
    
    HINT: Use cute.make_smem_tensor(smem_ptr, layout) where layout defines
          the shape and stride of the SMEM view.
    """
    # --- Step 1: Create SMEM layout ---
    # Row-major layout for (M, N) matrix
    # TODO: Create layout with shape (M, N) and stride (N, 1)
    
    # --- Step 2: Create SMEM tensor ---
    # TODO: smem_tensor = cute.make_smem_tensor(smem_ptr, layout)
    
    # --- Step 3: Initialize SMEM with thread 0 ---
    # Write a known pattern: element (i, j) = i + j
    # Only thread 0 writes to avoid races in this simple example
    # TODO: Use cute.thread_idx() to check if this is thread 0
    
    # --- Step 4: Read elements from SMEM ---
    # Store element (10, 5) in results[0]
    # Store element (20, 10) in results[1]
    # Store element (30, 15) in results[2]
    
    # --- Step 5: Verify the pattern ---
    # Expected: (10, 5) = 15, (20, 10) = 30, (30, 15) = 45
    results[3] = 10 + 5
    results[4] = 20 + 10
    results[5] = 30 + 15
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify SMEM tensor access.
    
    NCU PROFILING COMMAND:
    ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
        --set full --target-processes all \
        python ex02_smem_tensor_FILL_IN.py
    
    METRICS TO FOCUS ON:
    - l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum: Should be 0 for coalesced access
    - l1tex__t_bytes_pipe_lsu_mem_shared.sum: SMEM throughput
    """
    
    # Allocate shared memory via PyTorch (then get pointer)
    # In real kernels, SMEM is often declared statically in the kernel
    smem_torch = torch.zeros(M * N, dtype=torch.float32, device='cuda')
    smem_ptr = from_dlpack(smem_torch)
    
    # Allocate results tensor
    results_torch = torch.zeros(6, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel (1 block, 1 thread for simplicity)
    kernel_smem_tensor[1, 1](smem_ptr, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 02 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  SMEM Tensor: shape=({M}, {N}), {smem_size_bytes} bytes")
    print(f"\n  Results:")
    print(f"    Element (10, 5):        {results_cpu[0]}")
    print(f"    Element (20, 10):       {results_cpu[1]}")
    print(f"    Element (30, 15):       {results_cpu[2]}")
    print(f"    Expected (10, 5):       {results_cpu[3]}")
    print(f"    Expected (20, 10):      {results_cpu[4]}")
    print(f"    Expected (30, 15):      {results_cpu[5]}")
    
    # Verify
    passed = (
        results_cpu[0] == results_cpu[3] and
        results_cpu[1] == results_cpu[4] and
        results_cpu[2] == results_cpu[5]
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does SMEM allocation differ from GMEM in CuTe DSL?
# C2: What causes bank conflicts in SMEM and how do you avoid them?
# C3: In FlashAttention, what data is typically stored in SMEM?
# C4: Why is SMEM access ~100× faster than GMEM?

if __name__ == "__main__":
    run()
