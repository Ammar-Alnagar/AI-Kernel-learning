# Module 04 — TiledMMA: Tensor Core Compute Layer

## Concept Overview

TiledMMA is the **compute primitive** in CuTe. It defines how tensor core MMA (Matrix Multiply-Accumulate) operations are partitioned across threads. Combined with TiledCopy, this forms the complete GMEM→SMEM→RMEM→MMA pipeline.

### C++ → DSL Translation

| C++ 3.x | Python 4.x |
|---------|------------|
| `make_tiled_mma(MMA_Atom{}, atom_layout, val_layout)` | `cute.make_tiled_mma(mma_atom, atom_layout, val_layout)` |
| `thr_mma = tiled_mma.get_slice(thread_idx)` | `thr_mma = tiled_mma.get_slice(thread_idx)` |
| `gemm(tiled_mma, D, A, B, C)` | `cute.gemm(tiled_mma, D, A, B, C)` |
| `partition_fragment(...)` | `cute.make_rmem_tensor(shape, dtype)` |

### MMA Atoms

MMA atoms define the **elementary matrix multiply operation**:
- `MMA_Atom<MMAOp, A_DType, B_DType, C_DType>` — specifies MMA operation and dtypes
- Common atoms: `Mma_Sm80`, `Mma_Sm90`, `Mma_Sm100` (architecture-specific)
- Shapes: 16x16x16 (Volta/Ampere), 16x16x16 (Hopper), etc.

### Key API: `cute.gemm`

```python
# FP16 GEMM: D = A @ B + C
cute.gemm(tiled_mma, accum, a_frag, b_frag, accum)
```

---

## Exercises

| Exercise | Topic | Difficulty | Job Relevance |
|----------|-------|------------|---------------|
| 01 | MMA atom basics | [EASY] | Foundation for tensor core ops |
| 02 | TiledMMA setup | [MEDIUM] | Core GEMM building block |
| 03 | GEMM mainloop (QK^T style) | [HARD] | **FlashAttention core pattern** |
| 04 | Mixed precision (FP16 in, FP32 acc) | [HARD] | Production GEMM requirement |

---

## Why This Module Matters

### NVIDIA DL Software Engineer Interview
- TiledMMA is the **compute layer** in every CuTe kernel interview question
- Understanding MMA atom partitioning is essential for performance discussions

### FlashAttention / vLLM / TensorRT-LLM
- QK^T and PV matmuls in attention use TiledMMA
- Mixed precision (FP16 compute, FP32 accum) is standard for numerical stability

### Cerebras Wafer-Scale
- Matrix multiply is the core operation at wafer scale
- TiledMMA patterns map to Cerebras's matrix engines

---

## Prerequisites

You already have from CuTe C++ 3.x:
- MMA atom semantics (A/B/C dtypes, MMA shape)
- Thread/fragment partitioning
- `gemm` execution
- Accumulator clearing

**New in 4.x Python:**
- `make_rmem_tensor` for fragment creation
- Cleaner dtype specification
- `cute.clear(accum)` for accumulator reset

---

## Profiling Focus

```bash
# Ampere (SM80) - Tensor core utilization
ncu --metrics smsp__sass_thread_inst_executed_op_hadd_pred_on.sum,\
            smsp__sass_thread_inst_executed_op_fmul_sum,\
            tensor__pipe_tensor_op_hmma.sum \
    --set full --target-processes all \
    python ex03_gemm_mainloop_FILL_IN.py

# Hopper (SM90) - TMA + Tensor core overlap
ncu --metrics sm__inst_executed_pipe_tensor.sum,\
            tensor__pipe_tensor_op_hmma.sum \
    --set full --target-processes all \
    python ex04_mixed_precision_FILL_IN.py
```

---

**Next:** Open `ex01_mma_atom_FILL_IN.py`
