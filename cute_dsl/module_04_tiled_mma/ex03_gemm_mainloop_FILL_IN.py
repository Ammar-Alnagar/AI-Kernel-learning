"""
Module 04 — TiledMMA
Exercise 03 — GEMM Mainloop (QK^T Style)

CONCEPT BRIDGE (C++ → DSL):
  C++:  // Mainloop for Q @ K^T in FlashAttention
        for (int k = 0; k < K_BLOCKS; ++k) {
            copy(Q_tile[k], smem_Q);
            copy(K_tile[k], smem_K);
            gemm(smem_Q, smem_K, accum);
        }
  DSL:  # Same pattern in Python
        for k_block in range(K_BLOCKS):
            cute.copy(tiled_copy_Q, gmem_Q, smem_Q)
            cute.copy(tiled_copy_K, gmem_K, smem_K)
            cute.gemm(tiled_mma, accum, rmem_Q, rmem_K, accum)

WHAT YOU'RE BUILDING:
  The core mainloop pattern from FlashAttention's QK^T computation. This loops
  over the K (reduction) dimension in tiles, loading Q and K tiles and accumulating
  the attention scores. This is THE critical pattern for attention kernels.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Implement a tiled GEMM mainloop
  - Understand the load-compute overlap pattern
  - Apply this to FlashAttention's QK^T and PV matmuls

REQUIRED READING:
  - FlashAttention-2 paper: https://arxiv.org/abs/2307.08691 (Algorithm 1)
  - CUTLASS GEMM tutorial: https://nvidia.github.io/cutlass-dsl/tutorials/gemm.html
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: In QK^T with Q=(seq_q, head_dim) and K=(seq_k, head_dim), what is the
#     output shape and which dimension is the reduction dimension?
# Your answer:

# Q2: If head_dim=128 and we tile with tile_k=64, how many mainloop iterations?
# Your answer:

# Q3: Why do we accumulate in FP32 even when Q and K are FP16?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# QK^T dimensions for a small attention block
SEQ_Q = 64   # Query sequence length (block size)
SEQ_K = 64   # Key sequence length (block size)  
HEAD_DIM = 128  # Hidden dimension per head

# Tiling
TILE_K = 64  # Tile size along K dimension
K_BLOCKS = HEAD_DIM // TILE_K  # = 2 mainloop iterations

# Thread configuration (1 warp for this example)
NUM_THREADS = 32


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_gemm_mainloop(
    Q: cute.Tensor,
    K: cute.Tensor,
    S: cute.Tensor,  # Output: Q @ K^T
    results: cute.Tensor,
):
    """
    GEMM mainloop for QK^T computation (FlashAttention style).
    
    FILL IN [HARD]: Implement the tiled mainloop with load-compute pattern.
    
    HINT: The mainloop structure:
          for k_block in range(K_BLOCKS):
              1. Load Q tile from GMEM to RMEM
              2. Load K tile from GMEM to RMEM  
              3. MMA: accum += Q_tile @ K_tile^T
              
    Note: K needs to be accessed as K^T, so the layout is transposed.
    """
    # --- Step 1: Create MMA atom and TiledMMA ---
    # TODO: mma_atom = cute.MMA_atom(cute.Mma_Sm80, cutlass.float16, cutlass.float16, cutlass.float32)
    #       tiled_mma = cute.make_tiled_mma(mma_atom, (2, 4), (16, 16))
    
    # --- Step 2: Get thread index ---
    # TODO: tid = cute.thread_idx()
    #       thr_mma = tiled_mma.get_slice(tid)
    
    # --- Step 3: Initialize accumulator ---
    # TODO: accum = cute.make_rmem_tensor((16, 16), cutlass.float32)
    #       Clear accum to zeros
    
    # --- Step 4: Mainloop over K dimension ---
    # TODO: for k_block in range(K_BLOCKS):
    #           # Compute slice of K for this block
    #           k_start = k_block * TILE_K
    #           
    #           # Partition Q[:, k_start:k_start+TILE_K]
    #           # Partition K[:, k_start:k_start+TILE_K] (transposed)
    #           
    #           # Load to RMEM
    #           # Execute MMA: accum += Q @ K^T
    #       
    #       Store accum to S
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify GEMM mainloop.
    
    NCU PROFILING COMMAND:
    ncu --metrics tensor__pipe_tensor_op_hmma.sum,\
                smsp__sass_thread_inst_executed_op_hadd_pred_on.sum \
        --set full --target-processes all \
        python ex03_gemm_mainloop_FILL_IN.py
    """
    
    # Create Q and K matrices
    torch.manual_seed(42)
    Q_torch = torch.randn((SEQ_Q, HEAD_DIM), dtype=torch.float16, device='cuda')
    K_torch = torch.randn((SEQ_K, HEAD_DIM), dtype=torch.float16, device='cuda')
    
    # Reference: S = Q @ K^T
    S_ref = torch.matmul(Q_torch.float(), K_torch.float().T).cpu().numpy()
    
    # Output tensor
    S_torch = torch.zeros((SEQ_Q, SEQ_K), dtype=torch.float32, device='cuda')
    
    Q_cute = from_dlpack(Q_torch)
    K_cute = from_dlpack(K_torch)
    S_cute = from_dlpack(S_torch)
    
    # Results tensor
    results_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel
    kernel_gemm_mainloop[1, NUM_THREADS](Q_cute, K_cute, S_cute, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    S_cpu = S_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 04 — Exercise 03 Results")
    print("=" * 60)
    print(f"\n  QK^T GEMM Configuration:")
    print(f"    Q: ({SEQ_Q}, {HEAD_DIM})")
    print(f"    K: ({SEQ_K}, {HEAD_DIM})")
    print(f"    Output S: ({SEQ_Q}, {SEQ_K})")
    print(f"    K blocks: {K_BLOCKS}")
    print(f"\n  Results:")
    print(f"    S[0,0]:    {S_cpu[0, 0]:.4f} (ref: {S_ref[0, 0]:.4f})")
    print(f"    S[32,32]:  {S_cpu[32, 32]:.4f} (ref: {S_ref[32, 32]:.4f})")
    print(f"    S mean:    {S_cpu.mean():.4f} (ref: {S_ref.mean():.4f})")
    print(f"    S max:     {S_cpu.max():.4f} (ref: {S_ref.max():.4f})")
    
    # Verify
    max_diff = abs(S_cpu - S_ref).max()
    passed = max_diff < 1.0  # Allow some numerical difference for partial implementation
    
    print(f"\n  Max difference: {max_diff:.6f}")
    print(f"  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does the mainloop tile the K dimension?
# C2: In FlashAttention, what additional operations happen in the mainloop?
# C3: How would you extend this to handle causal masking?
# C4: What is the arithmetic intensity of this QK^T operation?

if __name__ == "__main__":
    run()
