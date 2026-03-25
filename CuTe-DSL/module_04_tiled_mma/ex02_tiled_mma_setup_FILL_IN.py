"""
Module 04 — TiledMMA
Exercise 02 — TiledMMA Setup

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto tiled_mma = make_tiled_mma(MMA_Atom{}, atom_layout, val_layout);
  DSL:  tiled_mma = cute.make_tiled_mma(mma_atom, atom_layout, val_layout)
  Key:  TiledMMA partitions MMA work across threads using atom/value layouts.

WHAT YOU'RE BUILDING:
  A complete TiledMMA object that partitions tensor core work across a warp.
  This is the core compute primitive used in GEMM and attention kernels. You'll
  set up the atom layout (how MMA atoms are arranged across threads) and value
  layout (how elements are assigned to each atom).

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create TiledMMA using make_tiled_mma
  - Understand atom layout (thread organization) and value layout
  - Execute tiled MMA with get_slice and cute.gemm

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_mma.html
  - FlashAttention-2 paper: https://arxiv.org/abs/2307.08691 (Section 3.2 on tiled matmul)
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: For a 16x16x16 MMA atom and a (2, 4) atom layout, what is the total
#     compute shape (M, N, K)?
# Your answer:

# Q2: How many threads are needed for a (2, 4) atom layout if each atom
#     uses 1 thread?
# Your answer:

# Q3: In FlashAttention's QK^T computation, what are the typical M, N, K dimensions?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# MMA atom for SM80 (FP16 × FP16 → FP32)
MMA_ATOM = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)

# Atom layout: (2, 4) = 8 atoms total
# This gives compute shape: (2*16, 4*16, 16) = (32, 64, 16)
ATOM_LAYOUT = (2, 4)

# Value layout: elements per atom (matches MMA atom output)
VAL_LAYOUT = (16, 16)  # Each atom produces 16x16 output

# Total compute shape
TOTAL_M = ATOM_LAYOUT[0] * 16  # = 32
TOTAL_N = ATOM_LAYOUT[1] * 16  # = 64
TOTAL_K = 16


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_tiled_mma(
    a_gmem: cute.Tensor,
    b_gmem: cute.Tensor,
    c_gmem: cute.Tensor,
    results: cute.Tensor,
):
    """
    Set up and execute a tiled MMA operation.
    
    FILL IN [MEDIUM]: Create TiledMMA and perform matrix multiply.
    
    HINT: tiled_mma = cute.make_tiled_mma(MMA_ATOM, ATOM_LAYOUT, VAL_LAYOUT)
          Then get thread slice: thr_mma = tiled_mma.get_slice(thread_idx)
          Partition fragments and execute: cute.gemm(...)
    """
    # --- Step 1: Create TiledMMA ---
    # TODO: tiled_mma = cute.make_tiled_mma(MMA_ATOM, ATOM_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Get thread-local MMA slice ---
    # TODO: tid = cute.thread_idx()
    #       thr_mma = tiled_mma.get_slice(tid)
    
    # --- Step 3: Partition input tensors for this thread ---
    # TODO: a_thread = thr_mma.partition_A(a_gmem)
    #       b_thread = thr_mma.partition_B(b_gmem)
    #       c_thread = thr_mma.partition_C(c_gmem)
    
    # --- Step 4: Create register fragments ---
    # TODO: rmem_a = cute.make_rmem_tensor_like(a_thread)
    #       rmem_b = cute.make_rmem_tensor_like(b_thread)
    #       rmem_c = cute.make_rmem_tensor_like(c_thread)
    
    # --- Step 5: Copy from GMEM to RMEM ---
    # TODO: Copy a_thread to rmem_a, b_thread to rmem_b
    
    # --- Step 6: Execute MMA ---
    # TODO: cute.gemm(tiled_mma, rmem_c, rmem_a, rmem_b, rmem_c)
    
    # --- Step 7: Store result back to GMEM ---
    # TODO: Copy rmem_c to c_thread
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify TiledMMA setup.
    
    NCU PROFILING COMMAND:
    ncu --metrics tensor__pipe_tensor_op_hmma.sum \
        --set full --target-processes all \
        python ex02_tiled_mma_setup_FILL_IN.py
    """
    
    # Create input matrices (small for verification)
    # A: (TOTAL_M, TOTAL_K) = (32, 16)
    # B: (TOTAL_K, TOTAL_N) = (16, 64)
    # C = A @ B should be (32, 64)
    
    torch.manual_seed(42)
    a_torch = torch.randn((TOTAL_M, TOTAL_K), dtype=torch.float16, device='cuda')
    b_torch = torch.randn((TOTAL_K, TOTAL_N), dtype=torch.float16, device='cuda')
    c_torch = torch.zeros((TOTAL_M, TOTAL_N), dtype=torch.float32, device='cuda')
    
    # Reference result
    c_ref = torch.matmul(a_torch.float(), b_torch.float()).cpu().numpy()
    
    a_cute = from_dlpack(a_torch)
    b_cute = from_dlpack(b_torch)
    c_cute = from_dlpack(c_torch)
    
    # Results tensor (store a few elements for verification)
    results_torch = torch.zeros(6, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel (8 threads = 8 atoms)
    kernel_tiled_mma[1, 8](a_cute, b_cute, c_cute, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    c_cpu = c_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 04 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  TiledMMA Configuration:")
    print(f"    Atom layout: {ATOM_LAYOUT}")
    print(f"    Compute shape: ({TOTAL_M}, {TOTAL_N}, {TOTAL_K})")
    print(f"    Threads: {ATOM_LAYOUT[0] * ATOM_LAYOUT[1]}")
    print(f"\n  Results:")
    print(f"    C[0,0]:   {c_cpu[0, 0]:.4f} (ref: {c_ref[0, 0]:.4f})")
    print(f"    C[16,32]: {c_cpu[16, 32]:.4f} (ref: {c_ref[16, 32]:.4f})")
    print(f"    C mean:   {c_cpu.mean():.4f} (ref: {c_ref.mean():.4f})")
    
    # Verify (allow small numerical differences)
    max_diff = abs(c_cpu - c_ref).max()
    passed = max_diff < 0.1
    
    print(f"\n  Max difference: {max_diff:.6f}")
    print(f"  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does TiledMMA partition work across threads?
# C2: What is the relationship between atom layout and compute shape?
# C3: In FlashAttention, how is TiledMMA used for QK^T and PV?
# C4: Why do we need register fragments for MMA operands?

if __name__ == "__main__":
    run()
