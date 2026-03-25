"""
Module 03 — TiledCopy
Exercise 02 — make_tiled_copy_tv Setup

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto tiled_copy = make_tiled_copy(Copy_Atom{}, thr_layout, val_layout);
  DSL:  tiled_copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)
  Key:  The _tv suffix makes the thread-value layout explicit (4.x API change).

WHAT YOU'RE BUILDING:
  A complete TiledCopy object that partitions copy work across threads. This is
  the core data movement primitive used in every CuTe kernel. You'll set up the
  thread layout, value layout, and copy atom, then execute a tiled copy from
  GMEM to SMEM.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create TiledCopy using make_tiled_copy_tv (4.x API)
  - Understand thread layout vs value layout partitioning
  - Execute tiled copies with get_slice and cute.copy

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_copy.html#make-tiled-copy-tv
  - Why _tv API: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute_tiled_copy.md

DEPRECATED API NOTE:
  make_tiled_copy is deprecated in 4.4. Always use make_tiled_copy_tv which makes
  the thread-value layout explicit and avoids implicit broadcast ambiguity.
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: For a thread layout of (4, 8) = 32 threads (1 warp), and a value layout
#     of (8, 4) = 32 elements per thread, how many total elements are copied?
# Your answer:

# Q2: What is the difference between thread layout and value layout?
# Your answer:

# Q3: Why was make_tiled_copy deprecated in favor of make_tiled_copy_tv?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Thread layout: (4, 8) = 32 threads (1 warp)
THR_LAYOUT = (4, 8)

# Value layout: elements per thread
VAL_LAYOUT = (8, 4)  # 32 elements per thread

# Total elements: 32 threads × 32 elements = 1024
TOTAL_ELEMENTS = 32 * 32

# Copy atom for FP32
COPY_ATOM = cute.Copy_atom(cute.UniversalCopy, cutlass.float32)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_tiled_copy_tv(
    src_gmem: cute.Tensor,
    dst_smem: cute.Pointer,
    results: cute.Tensor,
):
    """
    Set up and execute a tiled copy using make_tiled_copy_tv.
    
    FILL IN [MEDIUM]: Create TiledCopy and copy from GMEM to SMEM.
    
    HINT: tiled_copy = cute.make_tiled_copy_tv(COPY_ATOM, THR_LAYOUT, VAL_LAYOUT)
          Then get the thread slice: copy_slice = tiled_copy.get_slice(thread_idx)
          Finally: cute.copy(copy_slice, src, dst)
    """
    # --- Step 1: Create TiledCopy ---
    # TODO: tiled_copy = cute.make_tiled_copy_tv(COPY_ATOM, THR_LAYOUT, VAL_LAYOUT)
    
    # --- Step 2: Create SMEM tensor from pointer ---
    # The dst_smem needs a layout that matches the total copy size
    # TODO: dst_layout = cute.make_layout((TOTAL_ELEMENTS,), stride=(1,))
    #       dst_tensor = cute.make_smem_tensor(dst_smem, dst_layout)
    
    # --- Step 3: Get thread-local copy slice ---
    # TODO: tid = cute.thread_idx()
    #       copy_slice = tiled_copy.get_slice(tid)
    
    # --- Step 4: Partition source and destination for this thread ---
    # TODO: src_thread = copy_slice.partition_src(src_gmem)
    #       dst_thread = copy_slice.partition_dst(dst_tensor)
    
    # --- Step 5: Execute the copy ---
    # TODO: cute.copy(copy_slice, src_thread, dst_thread)
    
    # --- Step 6: Verify by reading first element from SMEM ---
    # Store dst_tensor[0] in results[0]
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify TiledCopy setup.
    
    NCU PROFILING COMMAND:
    ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
                l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
        --set full --target-processes all \
        python ex02_make_tiled_copy_tv_FILL_IN.py
    """
    
    # Create source tensor in GMEM
    src_torch = torch.arange(TOTAL_ELEMENTS, dtype=torch.float32, device='cuda')
    src_cute = from_dlpack(src_torch)
    
    # Allocate SMEM (via regular CUDA allocation for this exercise)
    dst_torch = torch.zeros(TOTAL_ELEMENTS, dtype=torch.float32, device='cuda')
    dst_ptr = from_dlpack(dst_torch)
    
    # Results tensor
    results_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel (1 block, 32 threads = 1 warp)
    kernel_tiled_copy_tv[1, 32](src_cute, dst_ptr, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    dst_cpu = dst_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 03 — Exercise 02 Results")
    print("=" * 60)
    print(f"\n  TiledCopy Configuration:")
    print(f"    Thread layout: {THR_LAYOUT} = {THR_LAYOUT[0] * THR_LAYOUT[1]} threads")
    print(f"    Value layout:  {VAL_LAYOUT} = {VAL_LAYOUT[0] * VAL_LAYOUT[1]} elements/thread")
    print(f"    Total elements: {TOTAL_ELEMENTS}")
    print(f"\n  Results:")
    print(f"    dst[0] from SMEM: {results_cpu[0]}")
    print(f"    Expected:         {results_cpu[1]}")
    print(f"    dst[100]:         {dst_cpu[100]}")
    print(f"    Expected:         {results_cpu[2]}")
    print(f"    dst[1023]:        {dst_cpu[1023]}")
    print(f"    Expected:         {results_cpu[3]}")
    
    # Verify
    passed = (
        results_cpu[0] == results_cpu[1] and
        dst_cpu[100] == results_cpu[2] and
        dst_cpu[1023] == results_cpu[3]
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: Why is make_tiled_copy_tv preferred over make_tiled_copy?
# C2: How does the thread layout map to actual thread indices?
# C3: In FlashAttention, what would be typical THR_LAYOUT and VAL_LAYOUT values?
# C4: What happens if the source tensor is smaller than the copy volume?

if __name__ == "__main__":
    run()
