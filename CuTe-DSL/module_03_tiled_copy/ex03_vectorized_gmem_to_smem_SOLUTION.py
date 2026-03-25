"""
Module 03 — TiledCopy
Exercise 03 — Vectorized GMEM to SMEM Copy

CONCEPT BRIDGE (C++ → DSL):
  C++:  using CopyAtom = SmemCopyAtom<uint128, float>;
        auto tiled_copy = make_tiled_copy_tv(CopyAtom{}, thr_layout, val_layout);
  DSL:  copy_atom = cute.Copy_atom(cute.SmemCopy, cutlass.float32, cutlass.float32, cute.b128)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
  Key:  Vectorized copies use wide load/store (128-bit) for maximum throughput.

WHAT YOU'RE BUILDING:
  A vectorized copy from GMEM to SMEM using 128-bit load/store instructions.
  This is the exact pattern used in FlashAttention for loading Q, K, V matrices
  from global memory into shared memory before attention computation. Vectorization
  is critical for achieving peak memory bandwidth.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create vectorized copy atoms with b128 width
  - Ensure proper alignment for vectorized access
  - Measure bandwidth improvement from vectorization

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_copy.html#vectorized-copies
  - FlashAttention-2 paper: https://arxiv.org/abs/2307.08691 (Section 3.1 on memory loading)
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What alignment is required for 128-bit (b128) vectorized loads?
# Your answer: 16-byte alignment (128 bits = 16 bytes)

# Q2: For FP32 data (32 bits), how many elements are loaded per b128 instruction?
# Your answer: 128 / 32 = 4 elements per instruction

# Q3: What is the theoretical memory bandwidth of an A100 GPU?
# Your answer: ~1555 GB/s (HBM2e)


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Matrix dimensions (must be aligned for vectorized access)
M, N = 256, 128  # 32K elements = 128 KB (FP32)
TOTAL = M * N

# Thread layout: 128 threads (4 warps)
THR_LAYOUT = (4, 32)

# Value layout: 256 elements per thread (enough for vectorization)
VAL_LAYOUT = (256,)

# Vectorized copy atom: b128 = 128-bit = 4 × FP32 elements
# Using SmemCopy for GMEM→SMEM transfer
COPY_ATOM = cute.Copy_atom(cute.SmemCopy, cutlass.float32, cutlass.float32, cute.b128)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_vectorized_copy(
    src_gmem: cute.Tensor,
    dst_smem: cute.Pointer,
    results: cute.Tensor,
):
    """
    Execute vectorized GMEM→SMEM copy.
    
    FILL IN [HARD]: Set up and execute vectorized tiled copy.
    
    HINT: The copy atom uses b128 vectorization. Ensure the source tensor
          is properly aligned (address divisible by 16 bytes).
          
    For FlashAttention-style loading:
    1. Create TiledCopy with vectorized atom
    2. Get thread slice
    3. Partition and copy
    """
    # --- Step 1: Create TiledCopy with vectorized atom ---
    tiled_copy = cute.make_tiled_copy_tv(COPY_ATOM, THR_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Create SMEM tensor ---
    dst_layout = cute.make_layout((M, N), stride=(N, 1))
    dst_tensor = cute.make_smem_tensor(dst_smem, dst_layout)
    
    # --- Step 3: Get thread slice and partition ---
    tid = cute.thread_idx()
    copy_slice = tiled_copy.get_slice(tid)
    src_thread = copy_slice.partition_src(src_gmem)
    dst_thread = copy_slice.partition_dst(dst_tensor)
    
    # --- Step 4: Execute vectorized copy ---
    cute.copy(copy_slice, src_thread, dst_thread)
    
    # --- Step 5: Verify copy (thread 0 only) ---
    if tid == 0:
        results[0] = dst_tensor[0, 0]
        results[1] = 0.0  # Expected
        results[2] = dst_tensor[100 // N, 100 % N]
        results[3] = 100.0  # Expected
        results[4] = dst_tensor[1000 // N, 1000 % N]
        results[5] = 1000.0  # Expected
        results[6] = dst_tensor[M - 1, N - 1]
        results[7] = float(TOTAL - 1)  # Expected
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and measure vectorized copy bandwidth.
    
    NCU PROFILING COMMAND:
    ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
                l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum,\
                dram__throughput.sum \
        --set full --target-processes all \
        python ex03_vectorized_gmem_to_smem_FILL_IN.py
    
    METRICS TO FOCUS ON:
    - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum: GMEM load bytes
    - dram__throughput.sum: DRAM utilization (% of peak)
    - l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum: Should be 0
    
    TARGET: >85% of peak memory bandwidth
    """
    
    # Create aligned source tensor in GMEM
    src_torch = torch.arange(TOTAL, dtype=torch.float32, device='cuda')
    src_cute = from_dlpack(src_torch)
    
    # Allocate SMEM
    dst_torch = torch.zeros(TOTAL, dtype=torch.float32, device='cuda')
    dst_ptr = from_dlpack(dst_torch)
    
    # Results tensor
    results_torch = torch.zeros(8, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel (1 block, 128 threads)
    kernel_vectorized_copy[1, 128](src_cute, dst_ptr, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    dst_cpu = dst_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 03 — Exercise 03 Results")
    print("=" * 60)
    print(f"\n  Vectorized Copy Configuration:")
    print(f"    Matrix: ({M}, {N}) = {TOTAL} elements ({TOTAL * 4 / 1024:.1f} KB)")
    print(f"    Threads: {THR_LAYOUT[0] * THR_LAYOUT[1]}")
    print(f"    Elements/thread: {VAL_LAYOUT[0]}")
    print(f"    Vectorization: b128 (4 × FP32)")
    print(f"\n  Verification:")
    print(f"    dst[0]:   {results_cpu[0]:.0f} (expected: {results_cpu[1]:.0f})")
    print(f"    dst[100]: {results_cpu[2]:.0f} (expected: {results_cpu[3]:.0f})")
    print(f"    dst[1000]: {results_cpu[4]:.0f} (expected: {results_cpu[5]:.0f})")
    print(f"    dst[{TOTAL-1}]: {results_cpu[6]:.0f} (expected: {results_cpu[7]:.0f})")
    
    # Verify
    passed = (
        results_cpu[0] == results_cpu[1] and
        results_cpu[2] == results_cpu[3] and
        results_cpu[4] == results_cpu[5] and
        results_cpu[6] == results_cpu[7]
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: What bandwidth did you achieve vs theoretical peak?
# C2: How does b128 vectorization compare to scalar copies?
# C3: In FlashAttention, where else would you use vectorized copies?
# C4: What happens if data is not 16-byte aligned for b128 loads?

if __name__ == "__main__":
    run()
