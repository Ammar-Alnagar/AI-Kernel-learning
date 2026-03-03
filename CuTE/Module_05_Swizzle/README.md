# Module 05: Swizzle — Bank-Conflict-Free Shared Memory

## What This Module Teaches

Swizzle is CuTe's abstraction for XOR-based shared memory address scrambling that eliminates bank conflicts during Tensor Core access. You will learn to apply `Swizzle<B, M, S>` to smem layouts, verify conflict-free access patterns, and understand the parameters that make FlashAttention-2 achieve peak smem bandwidth.

## Why Swizzle Matters for LLM Inference

**Job mapping:** NVIDIA Deep Learning Software Engineer (Inference) — "Optimize shared memory layout for Tensor Core MMA."

In FlashAttention-2:
1. K/V tiles are stored in swizzled smem layouts
2. Without swizzle: consecutive threads access consecutive addresses → bank conflicts
3. With swizzle: XOR scrambles the address → threads access different banks
4. Result: 32x bandwidth improvement during MMA fragment loading

This is the difference between 50% and 95% of peak Tensor Core throughput.

## Prerequisites

- Module 03: TiledCopy (gmem→smem)
- Module 04: TiledMMA (fragment access)
- GPU memory bank concepts (32 banks on Ada)

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_bank_conflicts_demo.cu | Visualize bank conflicts without swizzle | Naive smem access pattern |
| ex02_apply_swizzle.cu | `Swizzle<B, M, S>`, apply to smem layout | FlashAttention-2 K/V tile layout |
| ex03_verify_with_nsight.cu | Nsight Compute bank conflict metrics | Production profiling workflow |

## Exit Criteria

Before moving to Module 06, you must be able to:

1. Explain what causes shared memory bank conflicts
2. Apply `Swizzle<2, 3, 3>` to a smem layout and explain what each parameter does
3. Use Nsight Compute to verify zero bank conflicts in your kernel
4. Calculate the correct swizzle parameters for a given tile size

## Common Mistakes

1. **Wrong swizzle parameters:** The parameters `Swizzle<B, M, S>` depend on tile size and thread count. Using wrong parameters doesn't eliminate conflicts.

2. **Forgetting to apply swizzle to both load and store:** Both the TiledCopy (gmem→smem) and the MMA fragment access must use the same swizzled layout.

3. **Assuming swizzle is free:** Swizzle adds XOR instructions. The benefit (no bank conflicts) far outweighs the cost, but it's not zero-overhead.

4. **Not verifying with Nsight:** Always profile with `nvbankconflict` metrics to confirm zero conflicts. Don't assume your swizzle is correct.

---

Next: **ex01_bank_conflicts_demo.cu** — see bank conflicts in action.
