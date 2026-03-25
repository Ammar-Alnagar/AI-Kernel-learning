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
# Your answer: (2*16, 4*16, 16) = (32, 64, 16)

# Q2: How many threads are needed for a (2, 4) atom layout if each atom
#     uses 1 thread?
# Your answer: 2 * 4 = 8 threads

# Q3: In FlashAttention's QK^T computation, what are the typical M, N, K dimensions?
# Your answer: M = block_size_Q, N = block_size_K, K = head_dim


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# MMA atom for SM80 (FP16 × FP16 → FP32)
MMA_ATOM = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)

# Atom layout: (2, 4) = 8 atoms total
ATOM_LAYOUT = (2, 4)

# Value layout: elements per atom
VAL_LAYOUT = (16, 16)

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
    """
    # --- Step 1: Create TiledMMA ---
    tiled_mma = cute.make_tiled_mma(MMA_ATOM, ATOM_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Get thread-local MMA slice ---
    tid = cute.thread_idx()
    thr_mma = tiled_mma.get_slice(tid)
    
    # --- Step 3: Partition input tensors for this thread ---
    a_thread = thr_mma.partition_A(a_gmem)
    b_thread = thr_mma.partition_B(b_gmem)
    c_thread = thr_mma.partition_C(c_gmem)
    
    # --- Step 4: Create register fragments ---
    rmem_a = cute.make_rmem_tensor_like(a_thread)
    rmem_b = cute.make_rmem_tensor_like(b_thread)
    rmem_c = cute.make_rmem_tensor_like(c_thread)
    
    # --- Step 5: Copy from GMEM to RMEM ---
    for i in range(rmem_a.shape[0]):
        for j in range(rmem_a.shape[1]):
            rmem_a[i, j] = a_thread[i, j]
    
    for i in range(rmem_b.shape[0]):
        for j in range(rmem_b.shape[1]):
            rmem_b[i, j] = b_thread[i, j]
    
    # Clear accumulator
    for i in range(rmem_c.shape[0]):
        for j in range(rmem_c.shape[1]):
            rmem_c[i, j] = 0.0
    
    # --- Step 6: Execute MMA ---
    cute.gemm(tiled_mma, rmem_c, rmem_a, rmem_b, rmem_c)
    
    # --- Step 7: Store result back to GMEM ---
    for i in range(rmem_c.shape[0]):
        for j in range(rmem_c.shape[1]):
            c_thread[i, j] = rmem_c[i, j]
    
    # --- Step 8: Store verification results (thread 0) ---
    if tid == 0:
        results[0] = c_gmem[0, 0]
        results[1] = c_gmem[16, 32]
        results[2] = c_gmem.mean()
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify TiledMMA setup.
    """
    
    torch.manual_seed(42)
    a_torch = torch.randn((TOTAL_M, TOTAL_K), dtype=torch.float16, device='cuda')
    b_torch = torch.randn((TOTAL_K, TOTAL_N), dtype=torch.float16, device='cuda')
    c_torch = torch.zeros((TOTAL_M, TOTAL_N), dtype=torch.float32, device='cuda')
    
    c_ref = torch.matmul(a_torch.float(), b_torch.float()).cpu().numpy()
    
    a_cute = from_dlpack(a_torch)
    b_cute = from_dlpack(b_torch)
    c_cute = from_dlpack(c_torch)
    
    results_torch = torch.zeros(6, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    kernel_tiled_mma[1, 8](a_cute, b_cute, c_cute, results_cute)
    
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
