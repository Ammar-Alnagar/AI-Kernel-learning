# Module 03 — TiledCopy: Data Movement Layer

## Concept Overview

TiledCopy is the **data movement primitive** in CuTe. It defines how data moves between memory spaces (GMEM↔SMEM↔RMEM) using a copy atom that specifies the elementary copy operation. In CuTe 4.x, `make_tiled_copy_tv` is the canonical API (replacing the deprecated `make_tiled_copy`).

### C++ → DSL Translation

| C++ 3.x | Python 4.x |
|---------|------------|
| `make_tiled_copy(Copy_Atom{}, thr_layout, val_layout)` | `cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)` |
| `tiled_copy.get_slice(thread_idx)` | `tiled_copy.get_slice(thread_idx)` |
| `copy(tiled_copy, src, dst)` | `cute.copy(tiled_copy, src, dst)` |
| N/A (manual predication) | `cute.copy(atom, src, dst, pred=pred_tensor)` |

### Copy Atoms

Copy atoms define the **elementary copy operation**:
- `Copy_Atom<CopyOp, SrcDType, DstDType>` — specifies copy operation and dtypes
- Common atoms: `SmemCopyAtom`, `GmemCopyAtom`, `TmaCopyAtom` (SM90+)
- Vectorization: `b16`, `b32`, `b64`, `b128` for different vector widths

### Key API Change in 4.x

**Why `make_tiled_copy_tv` replaced `make_tiled_copy`:**
- `_tv` suffix makes the **thread-value layout** explicit
- Avoids implicit broadcast ambiguity in the older API
- More consistent with the value-partitioned tensor model

---

## Exercises

| Exercise | Topic | Difficulty | Job Relevance |
|----------|-------|------------|---------------|
| 01 | Copy atom basics | [EASY] | Foundation for all data movement |
| 02 | `make_tiled_copy_tv` setup | [MEDIUM] | **Core API for all tiled copies** |
| 03 | Vectorized GMEM→SMEM copy | [HARD] | FlashAttention QKV loading |
| 04 | TMA async copy (SM90+) | [HARD] | Hopper-specific optimization |

---

## Why This Module Matters

### NVIDIA DL Software Engineer Interview
- TiledCopy is the **data movement layer** in every CuTe kernel interview question
- Understanding copy atoms and vectorization is essential for performance discussions

### FlashAttention / vLLM / TensorRT-LLM
- QKV loading from GMEM→SMEM uses TiledCopy with vectorized atoms
- TMA (Tensor Memory Accelerator) on Hopper provides async copy with barrier sync

### Cerebras Wafer-Scale
- Data movement is the bottleneck at wafer scale
- TiledCopy patterns map to Cerebras's mesh-based data fabric

---

## Prerequisites

You already have from CuTe C++ 3.x:
- Copy atom semantics (Src/Dst dtype, copy operation)
- Thread/value layout partitioning
- `get_slice` for thread-local views
- `copy` execution

**New in 4.x Python:**
- `make_tiled_copy_tv` API (replaces `make_tiled_copy`)
- Predicated copy via `pred=` keyword
- Cleaner atom specification

---

## Profiling Focus

```bash
# Ampere (SM80) - Vectorized copies
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum,\
            l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    --set full --target-processes all \
    python ex03_vectorized_gmem_to_smem_FILL_IN.py

# Hopper (SM90) - TMA async copies
ncu --metrics sm__inst_executed_pipe_tensor.sum,\
            l2tex__t_bytes.sum \
    --set full --target-processes all \
    python ex04_tma_copy_hopper_FILL_IN.py
```

---

**Next:** Open `ex01_copy_atom_FILL_IN.py`
