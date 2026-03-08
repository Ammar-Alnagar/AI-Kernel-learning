"""
Module 04 — TiledMMA
Exercise 04 — Mixed Precision (FP16 In, FP32 Accum)

CONCEPT BRIDGE (C++ → DSL):
  C++:  using MmaAtom = MMA_Atom<Mma_Sm80, half_t, half_t, float>;
        // A, B are FP16; C (accumulator) is FP32
  DSL:  mma_atom = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
  Key:  Mixed precision uses FP16 for inputs (throughput) and FP32 for accumulation (stability).

WHAT YOU'RE BUILDING:
  Mixed precision GEMM — the standard for production attention kernels. FP16
  inputs provide 2× throughput over FP32, while FP32 accumulation maintains
  numerical stability for large reductions. This is exactly how FlashAttention
  and cuBLAS handle mixed precision.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Configure MMA atoms for mixed precision (FP16×FP16→FP32)
  - Handle dtype conversions between GMEM and RMEM
  - Understand when mixed precision is necessary

REQUIRED READING:
  - NVIDIA mixed precision guide: https://developer.nvidia.com/blog/accelerating-ai-training-with-tensor-cores-mixed-precision/
  - FlashAttention-2 paper: https://arxiv.org/abs/2307.08691 (Section 3.4 on numerical precision)
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What is the throughput advantage of FP16 tensor cores over FP32?
# Your answer:

# Q2: Why can't we use FP16 for the accumulator in large GEMMs?
# Your answer:

# Q3: What is the maximum value representable in FP16 before overflow?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# GEMM dimensions (small for verification)
M, N, K = 64, 64, 128

# MMA atom: FP16 inputs, FP32 accumulator
MMA_ATOM = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)

# Thread configuration (1 warp)
NUM_THREADS = 32


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_mixed_precision(
    A_fp16: cute.Tensor,
    B_fp16: cute.Tensor,
    C_fp32: cute.Tensor,
    results: cute.Tensor,
):
    """
    Mixed precision GEMM: C = A @ B with FP16 inputs and FP32 output.
    
    FILL IN [HARD]: Implement mixed precision GEMM with proper dtype handling.
    
    HINT: The MMA atom already specifies FP16×FP16→FP32.
          Key is ensuring A and B are loaded as FP16, C accumulates as FP32.
    """
    # --- Step 1: Create TiledMMA ---
    # TODO: tiled_mma = cute.make_tiled_mma(MMA_ATOM, (2, 4), (16, 16))
    
    # --- Step 2: Get thread slice ---
    # TODO: tid = cute.thread_idx()
    #       thr_mma = tiled_mma.get_slice(tid)
    
    # --- Step 3: Partition inputs ---
    # TODO: a_thread = thr_mma.partition_A(A_fp16)
    #       b_thread = thr_mma.partition_B(B_fp16)
    #       c_thread = thr_mma.partition_C(C_fp32)
    
    # --- Step 4: Create RMEM fragments with correct dtypes ---
    # TODO: rmem_a = cute.make_rmem_tensor(a_shape, cutlass.float16)
    #       rmem_b = cute.make_rmem_tensor(b_shape, cutlass.float16)
    #       rmem_c = cute.make_rmem_tensor(c_shape, cutlass.float32)
    
    # --- Step 5: Load and convert dtypes ---
    # Load A and B as FP16, initialize C as FP32 zeros
    
    # --- Step 6: Execute MMA ---
    # TODO: cute.gemm(tiled_mma, rmem_c, rmem_a, rmem_b, rmem_c)
    
    # --- Step 7: Store FP32 result ---
    # Store rmem_c to C_fp32 (already FP32)
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify mixed precision GEMM.
    
    NCU PROFILING COMMAND:
    ncu --metrics tensor__pipe_tensor_op_hmma.sum,\
                smsp__sass_thread_inst_executed_op_hadd_pred_on.sum \
        --set full --target-processes all \
        python ex04_mixed_precision_FILL_IN.py
    
    METRICS TO FOCUS ON:
    - tensor__pipe_tensor_op_hmma.sum: FP16 tensor core instructions
    - smsp__sass_thread_inst_executed_op_hadd_pred_on.sum: FP32 add instructions (accumulation)
    """
    
    # Create FP16 inputs
    torch.manual_seed(42)
    A_torch = torch.randn((M, K), dtype=torch.float16, device='cuda')
    B_torch = torch.randn((K, N), dtype=torch.float16, device='cuda')
    
    # Reference: FP32 GEMM for comparison
    C_ref = torch.matmul(A_torch.float(), B_torch.float()).cpu().numpy()
    
    # Output in FP32
    C_torch = torch.zeros((M, N), dtype=torch.float32, device='cuda')
    
    A_cute = from_dlpack(A_torch)
    B_cute = from_dlpack(B_torch)
    C_cute = from_dlpack(C_torch)
    
    # Results tensor
    results_torch = torch.zeros(6, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel
    kernel_mixed_precision[1, NUM_THREADS](A_cute, B_cute, C_cute, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    C_cpu = C_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 04 — Exercise 04 Results")
    print("=" * 60)
    print(f"\n  Mixed Precision GEMM:")
    print(f"    A: {A_torch.shape} ({A_torch.dtype})")
    print(f"    B: {B_torch.shape} ({B_torch.dtype})")
    print(f"    C: {C_torch.shape} ({C_torch.dtype})")
    print(f"\n  Results:")
    print(f"    C[0,0]:   {C_cpu[0, 0]:.6f} (ref: {C_ref[0, 0]:.6f})")
    print(f"    C[32,32]: {C_cpu[32, 32]:.6f} (ref: {C_ref[32, 32]:.6f})")
    print(f"    C mean:   {C_cpu.mean():.6f} (ref: {C_ref.mean():.6f})")
    print(f"    C max:    {C_cpu.max():.6f} (ref: {C_ref.max():.6f})")
    print(f"\n  Dtype verification:")
    print(f"    Input dtype:  {results_cpu[4]:.0f} (1 = FP16)")
    print(f"    Output dtype: {results_cpu[5]:.0f} (3 = FP32)")
    
    # Verify (allow small numerical differences due to FP16 rounding)
    max_diff = abs(C_cpu - C_ref).max()
    passed = max_diff < 0.5  # FP16 rounding can cause larger differences
    
    print(f"\n  Max difference: {max_diff:.6f}")
    print(f"  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: What precision loss do we accept with FP16 inputs?
# C2: When would you use FP8 instead of FP16?
# C3: How does mixed precision affect FlashAttention's numerical stability?
# C4: What is the TFLOPS advantage of FP16 tensor cores over FP32?

if __name__ == "__main__":
    run()
