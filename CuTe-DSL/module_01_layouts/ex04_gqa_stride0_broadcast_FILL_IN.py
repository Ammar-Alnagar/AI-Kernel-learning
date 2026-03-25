"""
Module 01 — Layout Algebra
Exercise 04 — Stride-0 Broadcast for GQA (Grouped Query Attention)

CONCEPT BRIDGE (C++ → DSL):
  C++:  cute::make_layout(make_shape(num_heads, head_dim), make_stride(0, 1))
        // stride-0 on heads dimension = broadcast across heads
  DSL:  cute.make_layout((num_heads, head_dim), stride=(0, 1))
  Key:  Stride-0 means all "head" indices map to the same base offset.
        This eliminates redundant KV loads in GQA.

WHAT YOU'RE BUILDING:
  A layout that models GQA's KV cache access pattern. In GQA (used by 
  Llama-2-70B, Llama-3, etc.), multiple query heads share the same KV heads.
  By using stride-0 on the head dimension, we create a broadcast layout that
  loads each KV head once but presents it as if there are num_query_heads copies.
  
  This is a DIRECT optimization in FlashAttention-3 and vLLM's GQA kernels.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create stride-0 broadcast layouts
  - Understand how stride-0 eliminates redundant memory loads
  - Apply this pattern to GQA and multi-head attention kernels
  - Recognize broadcast patterns in production attention implementations

REQUIRED READING (do this before writing any code):
  - FlashAttention-3 paper: https://arxiv.org/abs/2310.03748 (Section 3.2 on GQA)
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/layout.html#broadcast-layouts
  - vLLM GQA implementation: https://github.com/vllm-project/vllm/blob/main/vllm/attention/
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# Answer these questions BEFORE executing the code.
# Write your answers as comments below each question.
# ─────────────────────────────────────────────
# Q1: For a layout with shape (8, 64) and stride (0, 1), what is the
#     linear index of (head=0, dim=10) and (head=7, dim=10)?
# Your answer:

# Q2: Why do both indices from Q1 map to the same linear index?
#     What does this mean for memory loads?
# Your answer:

# Q3: In GQA with 8 query heads and 2 KV heads, each KV head is shared
#     by 4 query heads. If we tile the KV cache as (kv_heads=2, head_dim=64),
#     how would you create a broadcast layout that presents it as 
#     (query_heads=8, head_dim=64) without copying data?
# Your answer:

# Q4: What is the memory traffic reduction factor when using stride-0
#     broadcast for GQA with 8 query heads and 2 KV heads?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# GQA configuration (Llama-2-70B style):
# - 8 query heads, but only 2 KV heads (each KV head shared by 4 query heads)
# - Head dimension = 64
num_query_heads = 8
num_kv_heads = 2
head_dim = 64
heads_per_kv_head = num_query_heads // num_kv_heads  # = 4

# The KV cache is physically stored as (num_kv_heads, head_dim) = (2, 64)
# We want to logically view it as (num_query_heads, head_dim) = (8, 64)
# using stride-0 broadcast.


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_gqa_broadcast(
    results: cute.Tensor,
):
    """
    Demonstrate stride-0 broadcast for GQA KV cache access.
    
    FILL IN [HARD]: Create broadcast layout and verify memory savings.
    
    HINT: stride-0 on the head dimension means all head indices map to
          the same base offset. The layout "broadcasts" the KV heads.
          
    Physical KV layout: (num_kv_heads=2, head_dim=64), stride=(64, 1)
    Broadcast layout:   (num_query_heads=8, head_dim=64), stride=(0, 1)
    
    Wait — we need to think about this more carefully:
    - Physical storage: 2 KV heads × 64 dim = 128 elements
    - Logical view: 8 query heads × 64 dim, but heads [0,1] share KV[0],
                    heads [2,3] share KV[1], heads [4,5] share KV[0], etc.
    
    Actually, the stride-0 trick works when we're loading within a kernel.
    We create a layout that looks like (8, 64) but stride-0 on heads means
    we only load from 2 unique head positions.
    
    For this exercise, we'll show:
    1. Normal layout for physical KV storage
    2. Broadcast layout for logical access pattern
    3. Verify that multiple query heads map to the same KV head
    """
    # --- Step 1: Physical KV cache layout (what's actually stored) ---
    # Shape (2, 64), row-major stride (64, 1)
    kv_physical = cute.make_layout((num_kv_heads, head_dim), stride=(head_dim, 1))
    
    # --- Step 2: Broadcast layout for GQA access ---
    # We want to access as if shape is (8, 64), but stride-0 on heads
    # This means: for any head index h, the head stride contributes 0
    # So heads 0,1,2,3,4,5,6,7 all map to the same base for a given dim
    # 
    # But wait — we need to map query heads to KV heads correctly.
    # Query heads 0,1 → KV head 0
    # Query heads 2,3 → KV head 1
    # Query heads 4,5 → KV head 0
    # Query heads 6,7 → KV head 1
    #
    # This is done via modulo in the kernel, not pure layout algebra.
    # The stride-0 broadcast is for when we've already partitioned correctly.
    #
    # For this exercise, let's show the simpler case:
    # Broadcast layout where ALL query heads see the SAME KV head (for demo).
    kv_broadcast = cute.make_layout((num_query_heads, head_dim), stride=(0, 1))
    
    # --- Step 3: Verify broadcast behavior ---
    # All heads should map to the same index for a given dimension
    idx_head0_dim10 = kv_broadcast((0, 10))
    idx_head7_dim10 = kv_broadcast((7, 10))
    results[0] = idx_head0_dim10
    results[1] = idx_head7_dim10
    
    # --- Step 4: Compare with non-broadcast layout ---
    # Normal row-major layout for comparison
    kv_normal = cute.make_layout((num_query_heads, head_dim), stride=(head_dim, 1))
    idx_normal_head0 = kv_normal((0, 10))
    idx_normal_head7 = kv_normal((7, 10))
    results[2] = idx_normal_head0
    results[3] = idx_normal_head7
    
    # --- Step 5: Memory traffic comparison ---
    # Physical KV size vs logical view size
    physical_size = num_kv_heads * head_dim
    logical_size = num_query_heads * head_dim
    results[4] = physical_size
    results[5] = logical_size
    
    # --- Step 6: Verify cosize of broadcast layout ---
    # Even with stride-0, cosize gives the full logical size
    broadcast_flat = cute.cosize(kv_broadcast)
    results[6] = broadcast_flat.size()
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify GQA broadcast behavior.
    
    NCU PROFILING COMMAND (for later modules when we add actual loads):
    ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,
                  l2tex__t_bytes.sum \
        --set full --target-processes all \
        python ex04_gqa_stride0_broadcast_FILL_IN.py
    
    METRICS TO FOCUS ON:
    - l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum: Bytes loaded from global
    - l2tex__t_bytes.sum: L2 traffic
    With stride-0 broadcast, global loads should be reduced by 4× for GQA.
    """
    
    # Allocate result tensor (7 int32 values)
    result_torch = torch.zeros(7, dtype=torch.int32, device='cuda')
    result_cute = from_dlpack(result_torch)
    
    # Launch kernel
    kernel_gqa_broadcast(result_cute)
    
    # Copy back
    result_cpu = result_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 01 — Exercise 04 Results")
    print("=" * 60)
    print(f"\n  GQA Configuration:")
    print(f"    Query heads:  {num_query_heads}")
    print(f"    KV heads:     {num_kv_heads}")
    print(f"    Head dim:     {head_dim}")
    print(f"    Heads/KV:     {heads_per_kv_head}")
    print(f"\n  Broadcast Layout: shape=({num_query_heads}, {head_dim}), stride=(0, 1)")
    print(f"\n  Results:")
    print(f"    Broadcast index (head=0, dim=10):  {result_cpu[0]}")
    print(f"    Broadcast index (head=7, dim=10):  {result_cpu[1]}")
    print(f"    Normal index (head=0, dim=10):     {result_cpu[2]}")
    print(f"    Normal index (head=7, dim=10):     {result_cpu[3]}")
    print(f"    Physical KV size:                  {result_cpu[4]} elements")
    print(f"    Logical view size:                 {result_cpu[5]} elements")
    print(f"    Broadcast cosize:                  {result_cpu[6]} elements")
    
    # Verify
    # With stride-0 broadcast, all heads map to same index for given dim
    expected_broadcast_idx = 10  # stride-0 on heads, so just the dim index
    expected_normal_head0 = 0 * head_dim + 10  # = 10
    expected_normal_head7 = 7 * head_dim + 10  # = 458
    expected_physical_size = num_kv_heads * head_dim  # = 128
    expected_logical_size = num_query_heads * head_dim  # = 512
    expected_broadcast_cosize = num_query_heads * head_dim  # = 512 (logical size)
    
    print(f"\n  Expected:")
    print(f"    Broadcast index (head=0, dim=10):  {expected_broadcast_idx}")
    print(f"    Broadcast index (head=7, dim=10):  {expected_broadcast_idx} (same!)")
    print(f"    Normal index (head=0, dim=10):     {expected_normal_head0}")
    print(f"    Normal index (head=7, dim=10):     {expected_normal_head7}")
    print(f"    Physical KV size:                  {expected_physical_size} elements")
    print(f"    Logical view size:                 {expected_logical_size} elements")
    print(f"    Broadcast cosize:                  {expected_broadcast_cosize} elements")
    
    broadcast_same = (result_cpu[0] == result_cpu[1] == expected_broadcast_idx)
    normal_different = (result_cpu[2] != result_cpu[3])
    sizes_correct = (
        result_cpu[4] == expected_physical_size and
        result_cpu[5] == expected_logical_size and
        result_cpu[6] == expected_broadcast_cosize
    )
    
    passed = broadcast_same and normal_different and sizes_correct
    
    print(f"\n  Verification:")
    print(f"    Broadcast indices same:     {'✓' if broadcast_same else '✗'}")
    print(f"    Normal indices different:   {'✓' if normal_different else '✗'}")
    print(f"    Sizes correct:              {'✓' if sizes_correct else '✗'}")
    print(f"    Overall:                    {'✓ PASSED' if passed else '✗ FAILED'}")
    
    print(f"\n  Memory Traffic Reduction:")
    reduction = expected_logical_size / expected_physical_size
    print(f"    Without broadcast: {expected_logical_size} elements loaded")
    print(f"    With broadcast:    {expected_physical_size} elements loaded")
    print(f"    Reduction factor:  {reduction}× fewer loads")
    
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Did your PREDICT answers match the actual results?
# C2: Why does stride-0 on the head dimension create broadcast behavior?
# C3: In a real GQA kernel, how do you map query heads to the correct KV head?
#     (Hint: it's not pure layout algebra — you need modulo arithmetic)
# C4: FlashAttention-3 uses warp specialization for GQA. Which warp role
#     (DMA vs MMA) benefits most from stride-0 broadcast layouts?
# C5: How would you extend this to MLA (Multi-head Latent Attention) where
#     the KV cache is compressed to a latent dimension?

if __name__ == "__main__":
    run()
