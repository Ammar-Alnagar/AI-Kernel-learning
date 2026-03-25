"""
Module 04 — TiledMMA
Exercise 01 — MMA Atom Basics

CONCEPT BRIDGE (C++ → DSL):
  C++:  using MmaAtom = MMA_Atom<Mma_Sm80, float16_t, float16_t, float32_t>;
  DSL:  mma_atom = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
  Key:  MMA atoms define the elementary tensor core operation (A/B/C dtypes, MMA shape).

WHAT YOU'RE BUILDING:
  An MMA atom that specifies the tensor core operation for a specific architecture.
  MMA atoms are the building blocks of TiledMMA — they define the compute shape
  (e.g., 16x16x16), input dtypes (FP16), and accumulator dtype (FP32).

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create MMA atoms with specified architecture and dtypes
  - Understand the MMA shape (M, N, K) for tensor core operations
  - Distinguish between different MMA operations (SM80, SM90, SM100)

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_mma.html
  - NVIDIA Tensor Core docs: https://docs.nvidia.com/cuda/cublas/#tensor-core-operations
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What is the MMA shape for Ampere (SM80) tensor cores?
# Your answer:

# Q2: Why is the accumulator typically FP32 when inputs are FP16?
# Your answer:

# Q3: What is the difference between Mma_Sm80 and Mma_Sm90?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# MMA atom configuration for Ampere (SM80)
# FP16 inputs (A, B), FP32 accumulator (C)
MMA_OP = cute.Mma_Sm80
A_DTYPE = cutlass.float16
B_DTYPE = cutlass.float16
C_DTYPE = cutlass.float32

# MMA shape for SM80: 16x16x16
MMA_M, MMA_N, MMA_K = 16, 16, 16


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_mma_atom(
    a_frag: cute.Tensor,
    b_frag: cute.Tensor,
    c_frag: cute.Tensor,
    results: cute.Tensor,
):
    """
    Create and use an MMA atom for a small matrix multiply.
    
    FILL IN [EASY]: Create an MMA atom and perform a 16x16x16 MMA.
    
    HINT: mma_atom = cute.MMA_atom(MMA_OP, A_DTYPE, B_DTYPE, C_DTYPE)
          Then use cute.gemm(mma_atom, c, a, b, c) for the actual MMA.
    """
    # --- Step 1: Create MMA atom ---
    # TODO: mma_atom = cute.MMA_atom(MMA_OP, A_DTYPE, B_DTYPE, C_DTYPE)
    
    # --- Step 2: Initialize fragments ---
    # A fragment: (MMA_M, MMA_K) = (16, 16)
    # B fragment: (MMA_K, MMA_N) = (16, 16)
    # C fragment: (MMA_M, MMA_N) = (16, 16)
    # Initialize A and B with 1.0, C with 0.0
    
    # --- Step 3: Execute MMA: C = A @ B + C ---
    # TODO: cute.gemm(mma_atom, c_frag, a_frag, b_frag, c_frag)
    
    # --- Step 4: Verify result ---
    # For A=1, B=1, C should be all 16.0 (sum of 16 products of 1*1)
    # Store C[0, 0] in results[0]
    # Store expected value (16.0) in results[1]
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify MMA atom behavior.
    
    NCU PROFILING COMMAND:
    ncu --metrics tensor__pipe_tensor_op_hmma.sum \
        --set full --target-processes all \
        python ex01_mma_atom_FILL_IN.py
    """
    
    # Create fragments in registers (simulated via CUDA memory)
    a_torch = torch.ones((MMA_M, MMA_K), dtype=torch.float16, device='cuda')
    b_torch = torch.ones((MMA_K, MMA_N), dtype=torch.float16, device='cuda')
    c_torch = torch.zeros((MMA_M, MMA_N), dtype=torch.float32, device='cuda')
    
    a_cute = from_dlpack(a_torch)
    b_cute = from_dlpack(b_torch)
    c_cute = from_dlpack(c_torch)
    
    # Results tensor
    results_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel (1 thread for simplicity)
    kernel_mma_atom[1, 1](a_cute, b_cute, c_cute, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    c_cpu = c_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 04 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  MMA Atom: {MMA_OP}, {A_DTYPE} × {B_DTYPE} → {C_DTYPE}")
    print(f"  MMA Shape: ({MMA_M}, {MMA_N}, {MMA_K})")
    print(f"\n  Input: A=ones(16,16), B=ones(16,16), C=zeros(16,16)")
    print(f"  Output: C = A @ B + C")
    print(f"\n  Results:")
    print(f"    C[0,0]: {results_cpu[0]:.1f} (expected: {results_cpu[1]:.1f})")
    print(f"    C mean: {c_cpu.mean():.1f} (expected: 16.0)")
    
    # Verify: each element of C should be sum of 16 products of 1*1 = 16
    expected = 16.0
    passed = (
        abs(results_cpu[0] - expected) < 0.1 and
        abs(c_cpu.mean() - expected) < 0.1
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: What is the purpose of an MMA atom vs a full TiledMMA?
# C2: Why is FP32 accumulation important for numerical stability?
# C3: How does the MMA shape relate to tensor core hardware?
# C4: In FlashAttention, where are MMA atoms used?

if __name__ == "__main__":
    run()
