# Module 06: Pipeline — Double Buffering and Load/Compute Overlap

## What This Module Teaches

Pipeline is the pattern that hides memory latency by overlapping load and compute. You will learn double buffering (2-stage pipeline), `cp_async_wait` synchronization, and the prologue/mainloop/epilogue structure used in FlashAttention-2 to achieve peak throughput.

## Why Pipeline Matters for LLM Inference

**Job mapping:** NVIDIA Deep Learning Software Engineer (Inference) — "Implement pipelined FlashAttention-2 with load/compute overlap."

In FlashAttention-2:
1. **Prologue:** Load first K/V tile into buffer 0
2. **Mainloop:** While computing QK^T with buffer 0, load next K/V tile into buffer 1
3. **Epilogue:** Finish last tile computation
4. Result: Memory latency is hidden behind computation

This is the difference between latency-bound and compute-bound execution.

## Prerequisites

- Module 03: TiledCopy (cp.async, cp_async_fence, cp_async_wait)
- Module 04: TiledMMA (GEMM execution)
- Module 05: Swizzle (bank-conflict-free smem)

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_double_buffer.cu | 2-stage pipeline, ping-pong buffers | FlashAttention-2 K/V double buffering |
| ex02_pipelined_gemm.cu | Full pipeline with GEMM compute | QK^T with overlapped K/V load |
| ex03_async_mma_overlap.cu | `cp_async_wait<1>`, async MMA | Production FlashAttention-2 pattern |

## Exit Criteria

Before moving to Projects, you must be able to:

1. Implement a 2-stage pipeline with prologue/mainloop/epilogue
2. Use `cp_async_wait<1>` to wait for "previous" copy while issuing next
3. Explain why FlashAttention-2 needs double buffering (not single buffering)
4. Calculate the tile size where pipeline benefit becomes significant

## Common Mistakes

1. **Wrong cp_async_wait argument:** `cp_async_wait<N>` waits until N copies remain pending. For double buffering, use `cp_async_wait<1>` (wait for previous, keep 1 pending).

2. **Missing __syncthreads():** After cp_async_wait completes, you still need `__syncthreads()` before threads can safely use the loaded data.

3. **Pipeline bubble in prologue:** The prologue must load the first tile BEFORE the mainloop starts. Starting mainloop without data causes stalls.

4. **Not handling the last tile:** The epilogue must process the final tile after the mainloop ends. Forgetting this loses the last output.

---

Next: **ex01_double_buffer.cu** — your first double-buffered pipeline.
