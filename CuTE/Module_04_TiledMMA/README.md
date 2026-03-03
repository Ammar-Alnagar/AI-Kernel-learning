# Module 04: TiledMMA — Warp-Level Tensor Core GEMM

## What This Module Teaches

TiledMMA is CuTe's abstraction for Tensor Core matrix multiply-accumulate operations. You will learn to construct MMA atoms for sm_89 (Ada Lovelace), partition fragments across warps, and execute FP16 GEMMs with FP32 accumulation — the core operation in FlashAttention-2's QK^T and PV computation.

## Why TiledMMA Matters for LLM Inference

**Job mapping:** NVIDIA Deep Learning Software Engineer (Inference) — "Implement TiledMMA for FlashAttention-2 QK^T and PV GEMMs."

In FlashAttention-2:
1. QK^T = TiledMMA(Q_tile, transpose(K_tile), {}) — compute attention scores
2. PV = TiledMMA(softmax(QK^T), V_tile, {}) — compute output
3. Fragments are partitioned across warps for parallel execution
4. FP16 inputs with FP32 accumulation for numerical stability

This is where 90% of the FLOPs happen in attention.

## Prerequisites

- Module 01: Layouts (make_layout, logical_divide)
- Module 02: Tensors (make_tensor, local_tile)
- Module 03: TiledCopy (gmem→smem, cp.async)
- Tensor Core basics (wmma, mma.sync)

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_mma_atom_setup.cu | MMA atoms, sm_89 instructions | Tensor Core configuration |
| ex02_tiled_gemm.cu | `make_tiled_mma`, `gemm()` call | QK^T attention score GEMM |
| ex03_fragment_layout.cu | `partition_fragment_A/B/C`, warp distribution | FlashAttention-2 warp specialization |
| ex04_mixed_precision_fp16.cu | FP16 inputs, FP32 accumulator | Numerical stability for softmax |

## Exit Criteria

Before moving to Module 05, you must be able to:

1. Construct a TiledMMA operator for sm_89: `make_tiled_mma(MMA_Atom, layout_warp)`
2. Partition fragments: `partition_fragment_A(tiled_mma, A_layout)`
3. Execute a GEMM: `gemm(tiled_mma, A_frag, B_frag, C_frag)`
4. Explain why FlashAttention-2 uses FP16 inputs with FP32 accumulation

## Common Mistakes

1. **Wrong fragment layout:** MMA fragments have a specific layout determined by the MMA atom. Using the wrong layout causes incorrect results or crashes.

2. **Missing warp synchronization:** After `gemm()`, threads must synchronize before using results. `__syncthreads()` or `warp_group_sync()` is required.

3. **Confusing row-major and column-major:** Tensor Core MMA uses specific layouts for each operand. A is typically row-major, B is column-major, C matches output layout.

4. **Not using FP32 accumulation:** FP16 accumulation loses precision for large GEMMs. Always use FP32 for the accumulator (C fragment).

---

Next: **ex01_mma_atom_setup.cu** — your first Tensor Core MMA atom.
