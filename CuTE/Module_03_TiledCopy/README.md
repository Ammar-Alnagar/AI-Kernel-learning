# Module 03: TiledCopy — Efficient Memory Movement

## What This Module Teaches

TiledCopy is CuTe's abstraction for high-bandwidth memory transfers. You will learn to copy data from global memory to shared memory using vectorized 128-bit loads, async copy (cp.async), and pipelined load/compute overlap — the exact patterns used in FlashAttention-2 and CUTLASS.

## Why TiledCopy Matters for LLM Inference

**Job mapping:** NVIDIA Deep Learning Software Engineer (Inference) — "Implement TiledCopy with cp.async for FlashAttention-2 K/V loading."

In FlashAttention-2:
1. K/V tiles are loaded from gmem to smem using TiledCopy
2. 128-bit vectorized loads (float4) maximize memory bandwidth
3. cp.async allows load/compute overlap — while computing QK^T, the next K/V tile loads
4. Proper tiling ensures coalesced access and avoids bank conflicts

This is the difference between 50% and 90% of peak memory bandwidth.

## Prerequisites

- Module 01: Layouts (make_layout, logical_divide)
- Module 02: Tensors (make_tensor, local_tile)
- CUDA shared memory (__shared__, __syncthreads)
- Memory coalescing concepts

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_basic_copy.cu | `make_tiled_copy`, Copy atoms, `copy()` | Basic gmem→smem K/V load |
| ex02_vectorized_128bit.cu | float4/128-bit vectorized loads | Coalesced attention input loading |
| ex03_gmem_to_smem.cu | Shared memory tensors, smem layout | K/V tile storage for MMA |
| ex04_async_copy_pipeline.cu | `cp.async`, `cp_async_fence`, `cp_async_wait` | FlashAttention-2 load/compute overlap |

## Exit Criteria

Before moving to Module 04, you must be able to:

1. Construct a TiledCopy operator: `make_tiled_copy(Copy_Atom, layout_thread, layout_smem)`
2. Explain the difference between 32-bit and 128-bit copy atoms
3. Write a gmem→smem copy that achieves >80% of peak memory bandwidth
4. Use cp.async_wait to synchronize async copies before using data

## Common Mistakes

1. **Uncoalesced access:** TiledCopy must be configured so consecutive threads access consecutive memory addresses. Otherwise bandwidth drops dramatically.

2. **Missing cp_async_fence:** After issuing cp.async copies, you must call cp_async_fence to commit them. Without this, copies never execute.

3. **Using data before cp_async_wait completes:** cp.async_wait<N> waits until N copies remain pending. Using data before wait completes causes race conditions.

4. **Wrong shared memory size:** Dynamic shared memory must be explicitly specified in kernel launch: `kernel<<<grid, block, smem_size>>>`.

---

Next: **ex01_basic_copy.cu** — your first TiledCopy.
