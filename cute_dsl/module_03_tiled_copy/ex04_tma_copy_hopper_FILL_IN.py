"""
Module 03 — TiledCopy
Exercise 04 — TMA Async Copy (Hopper SM90+)

CONCEPT BRIDGE (C++ → DSL):
  C++:  using TmaCopyAtom = TmaCopyAtom<uint128, float>;
        auto tiled_copy = make_tiled_copy_tv(TmaCopyAtom{}, thr_layout, val_layout);
  DSL:  copy_atom = cute.Copy_atom(cute.TmaCopy, cutlass.float32, cute.b128)
        tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
  Key:  TMA (Tensor Memory Accelerator) is Hopper's async copy engine with
        hardware barrier synchronization.

WHAT YOU'RE BUILDING:
  TMA async copy for Hopper GPUs (SM90+). TMA provides hardware-accelerated
  async memory copies with barrier synchronization — essential for warp-specialized
  pipelines in FlashAttention-3. This exercise shows the TMA setup pattern.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create TMA copy atoms for Hopper
  - Understand async copy with barrier synchronization
  - Recognize TMA's role in warp-specialized pipelines

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_copy.html#tma-copy
  - Hopper TMA docs: https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html#tensor-memory-accelerator
  - FlashAttention-3 paper: https://arxiv.org/abs/2310.03748 (Section 3.3 on TMA)

NOTE: This exercise requires Hopper (SM90) or later hardware.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What is the key advantage of TMA over regular async copy?
# Your answer:

# Q2: How does TMA integrate with barrier synchronization?
# Your answer:

# Q3: In FlashAttention-3's warp specialization, which warp role uses TMA?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Check if we're on Hopper
def is_hopper():
    try:
        import torch
        if torch.cuda.is_available():
            cc = torch.cuda.get_device_capability(0)
            return cc[0] >= 9
    except:
        pass
    return False

# Matrix dimensions
M, N = 256, 128
TOTAL = M * N

# Thread layout for TMA: typically 128 threads (4 warps) for DMA warps
THR_LAYOUT = (4, 32)

# Value layout: elements per thread
VAL_LAYOUT = (256,)

# TMA copy atom: async copy with barrier
# TMA uses special copy operation and includes barrier ID
COPY_ATOM = cute.Copy_atom(cute.TmaCopy, cutlass.float32, cute.b128)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_tma_copy(
    src_gmem: cute.Tensor,
    dst_smem: cute.Pointer,
    barrier: cute.Barrier,
    results: cute.Tensor,
):
    """
    Execute TMA async copy with barrier synchronization.
    
    FILL IN [HARD]: Set up TMA copy and synchronize with barrier.
    
    HINT: TMA copy is async — you must use barrier.sync() to wait for completion.
          The barrier is initialized with the number of threads participating.
          
    TMA pattern for warp-specialized pipelines:
    1. DMA warps issue TMA copies
    2. Barrier tracks copy completion
    3. MMA warps wait on barrier before consuming data
    """
    # --- Step 1: Create TMA TiledCopy ---
    # TODO: tiled_copy = cute.make_tiled_copy_tv(COPY_ATOM, THR_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Create SMEM tensor ---
    # TODO: dst_layout = cute.make_layout((M, N), stride=(N, 1))
    #       dst_tensor = cute.make_smem_tensor(dst_smem, dst_layout)
    
    # --- Step 3: Get thread slice ---
    # TODO: tid = cute.thread_idx()
    #       copy_slice = tiled_copy.get_slice(tid)
    
    # --- Step 4: Issue async TMA copy ---
    # TODO: src_thread = copy_slice.partition_src(src_gmem)
    #       dst_thread = copy_slice.partition_dst(dst_tensor)
    #       cute.copy(copy_slice, src_thread, dst_thread, barrier=barrier)
    
    # --- Step 5: Wait for copy completion ---
    # TODO: barrier.sync()
    
    # --- Step 6: Verify (thread 0) ---
    # Store results after barrier sync (data is ready)
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify TMA async copy.
    
    NCU PROFILING COMMAND (Hopper only):
    ncu --metrics sm__inst_executed_pipe_tensor.sum,\
                l2tex__t_bytes.sum,\
                gpu__time_duration.sum \
        --set full --target-processes all \
        python ex04_tma_copy_hopper_FILL_IN.py
    
    METRICS TO FOCUS ON:
    - sm__inst_executed_pipe_tensor.sum: Tensor core instructions (TMA)
    - l2tex__t_bytes.sum: L2 traffic
    - gpu__time_duration.sum: Kernel duration
    
    NOTE: TMA requires Hopper (SM90) or later.
    """
    
    if not is_hopper():
        print("\n  ⚠️  TMA requires Hopper (SM90) or later GPU.")
        print("  Skipping TMA exercise — reviewing code only.\n")
        print("  On Hopper, this would execute async TMA copies with barrier sync.")
        return True
    
    # Create source tensor
    src_torch = torch.arange(TOTAL, dtype=torch.float32, device='cuda')
    src_cute = from_dlpack(src_torch)
    
    # Allocate SMEM
    dst_torch = torch.zeros(TOTAL, dtype=torch.float32, device='cuda')
    dst_ptr = from_dlpack(dst_torch)
    
    # Create barrier (initialized to 0, expects 128 threads)
    # In CuTe DSL, barriers are created via the runtime
    barrier = cute.Barrier(128)
    
    # Results tensor
    results_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel (1 block, 128 threads)
    kernel_tma_copy[1, 128](src_cute, dst_ptr, barrier, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    dst_cpu = dst_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 03 — Exercise 04 Results (Hopper TMA)")
    print("=" * 60)
    print(f"\n  TMA Copy Configuration:")
    print(f"    Matrix: ({M}, {N}) = {TOTAL} elements")
    print(f"    Threads: {THR_LAYOUT[0] * THR_LAYOUT[1]}")
    print(f"    Vectorization: b128 (4 × FP32)")
    print(f"\n  Verification:")
    print(f"    dst[0]:   {results_cpu[0]:.0f} (expected: {results_cpu[1]:.0f})")
    print(f"    dst[{TOTAL-1}]: {results_cpu[2]:.0f} (expected: {results_cpu[3]:.0f})")
    
    # Verify
    passed = (
        results_cpu[0] == results_cpu[1] and
        results_cpu[2] == results_cpu[3]
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does TMA differ from regular async copy?
# C2: What is the role of the barrier in TMA synchronization?
# C3: In FlashAttention-3, how are DMA and MMA warps coordinated?
# C4: What are the alignment requirements for TMA copies?

if __name__ == "__main__":
    run()
