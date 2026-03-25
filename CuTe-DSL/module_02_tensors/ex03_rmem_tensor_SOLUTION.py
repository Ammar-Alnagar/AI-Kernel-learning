"""
Module 02 — Tensors
Exercise 03 — RMEM Tensor (Register Fragments)

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto rmem_frag = thr_mma.partition_fragment(...);  // C++ 3.x
  DSL:  rmem_tensor = cute.make_rmem_tensor(shape, dtype)  // Python 4.x
  Key:  RMEM tensors are thread-local register files for MMA operands.
        The 4.x API simplifies fragment creation with explicit shape/dtype.

WHAT YOU'RE BUILDING:
  Register memory tensors that hold MMA operation fragments. In the GMEM→SMEM→RMEM
  pipeline, RMEM is the final stage where actual tensor core operations happen.
  You'll create register tensors, initialize them, and understand their role in
  the MMA pipeline.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create RMEM tensors with explicit shape and dtype
  - Understand register file pressure and capacity limits
  - Use make_rmem_tensor_like for creating matching operand fragments

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tensor.html#register-memory
  - NVIDIA Tensor Core docs: https://docs.nvidia.com/cuda/cublas/#tensor-core-operations
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What is the typical register file size per SM on Ampere (SM80)?
# Your answer: 64K registers per SM (255 registers per thread max)

# Q2: For a 16x16x16 MMA operation (FP16), how many register elements does
#     each thread need for the A fragment, B fragment, and C fragment?
# Your answer: Depends on thread layout. For a typical 8x8 thread block:
#              A: 8x16=128, B: 16x8=128, C: 8x8=64 elements

# Q3: Why are RMEM tensors not backed by a pointer like GMEM/SMEM tensors?
# Your answer: Registers are implicit per-thread storage. No pointer needed
#              because each thread has its own register file.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# MMA fragment shape for a 16x16x16 tile
# For a single thread's fragment in a warp-level MMA:
# - A fragment: (8, 16) for FP16
# - B fragment: (16, 8) for FP16  
# - C fragment (accumulator): (8, 8) for FP32
frag_m_shape = (8, 16)
frag_n_shape = (16, 8)
frag_c_shape = (8, 8)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_rmem_tensor(
    results: cute.Tensor,
):
    """
    Create and manipulate RMEM tensors for MMA fragments.
    
    FILL IN [MEDIUM]: Create register tensors and perform fragment operations.
    
    HINT: RMEM tensors are created with cute.make_rmem_tensor(shape, dtype).
          They don't need a pointer — registers are implicit.
    """
    # --- Step 1: Create A fragment (FP16) ---
    rmem_A = cute.make_rmem_tensor(frag_m_shape, cutlass.float16)
    
    # --- Step 2: Create B fragment (FP16) ---
    rmem_B = cute.make_rmem_tensor(frag_n_shape, cutlass.float16)
    
    # --- Step 3: Create C fragment (FP32 accumulator) ---
    rmem_C = cute.make_rmem_tensor(frag_c_shape, cutlass.float32)
    
    # --- Step 4: Initialize fragments with known values ---
    # Initialize A with all 1.0, B with all 2.0, C with all 0.0
    for i in range(frag_m_shape[0]):
        for j in range(frag_m_shape[1]):
            rmem_A((i, j)) = 1.0
    
    for i in range(frag_n_shape[0]):
        for j in range(frag_n_shape[1]):
            rmem_B((i, j)) = 2.0
    
    for i in range(frag_c_shape[0]):
        for j in range(frag_c_shape[1]):
            rmem_C((i, j)) = 0.0
    
    # --- Step 5: Simulate MMA: C = A @ B + C ---
    # Compute C[0,0] = sum(A[0,:] * B[:,0])
    accum = 0.0
    for k in range(frag_m_shape[1]):  # = 16
        accum += rmem_A((0, k)) * rmem_B((k, 0))
    rmem_C((0, 0)) = accum
    
    # --- Step 6: Store fragment shapes in results ---
    results[0] = float(frag_m_shape[0] * frag_m_shape[1])  # A size
    results[1] = float(frag_n_shape[0] * frag_n_shape[1])  # B size
    results[2] = float(frag_c_shape[0] * frag_c_shape[1])  # C size
    results[3] = rmem_C((0, 0))  # C[0,0] computed
    results[4] = 32.0  # Expected: 16 elements × 1.0 × 2.0 = 32.0
    
    # Sum of all C (only [0,0] is non-zero)
    results[5] = rmem_C((0, 0))
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify RMEM tensor creation.
    
    NCU PROFILING COMMAND:
    ncu --metrics smsp__thread_reg_alloc.sum \
        --set full --target-processes all \
        python ex03_rmem_tensor_FILL_IN.py
    
    METRICS TO FOCUS ON:
    - smsp__thread_reg_alloc.sum: Registers allocated per thread
    - gpr_utilization: Register file utilization
    """
    
    # Allocate results tensor
    results_torch = torch.zeros(6, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel
    kernel_rmem_tensor[1, 128](results_cute)  # 128 threads = 4 warps
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 02 — Exercise 03 Results")
    print("=" * 60)
    print(f"\n  RMEM Fragments:")
    print(f"    A fragment shape: {frag_m_shape}")
    print(f"    B fragment shape: {frag_n_shape}")
    print(f"    C fragment shape: {frag_c_shape}")
    print(f"\n  Results:")
    print(f"    A fragment size:      {results_cpu[0]}")
    print(f"    B fragment size:      {results_cpu[1]}")
    print(f"    C fragment size:      {results_cpu[2]}")
    print(f"    C[0,0] (computed):    {results_cpu[3]}")
    print(f"    C[0,0] (expected):    {results_cpu[4]}")
    print(f"    Sum of all C:         {results_cpu[5]}")
    
    # Verify sizes
    expected_a_size = 8 * 16  # = 128
    expected_b_size = 16 * 8  # = 128
    expected_c_size = 8 * 8   # = 64
    # C[0,0] = sum(A[0,:] * B[:,0]) = sum(1.0 * 2.0 for 16 elements) = 32.0
    expected_c00 = 16 * 2.0  # = 32.0
    
    passed = (
        results_cpu[0] == expected_a_size and
        results_cpu[1] == expected_b_size and
        results_cpu[2] == expected_c_size and
        abs(results_cpu[3] - expected_c00) < 0.01
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does RMEM tensor creation differ from GMEM/SMEM?
# C2: What happens if your kernel exceeds the register file capacity?
# C3: In a full GEMM kernel, how many RMEM fragments does each thread need?
# C4: Why is the C accumulator FP32 when A and B are FP16?

if __name__ == "__main__":
    run()
