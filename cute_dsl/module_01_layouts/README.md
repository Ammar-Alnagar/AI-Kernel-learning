# Module 01 — Layout Algebra in CuTe DSL

## Concept Overview

Layouts are the foundation of CuTe. A layout defines a **bijective mapping** from logical coordinates to linear indices. Everything in CuTe — tensors, tiles, thread assignments — builds on this algebra.

### C++ → DSL Translation

| C++ 3.x | Python 4.x |
|---------|------------|
| `make_layout(make_shape(M, N), make_stride(N, 1))` | `cute.make_layout((M, N), stride=(N, 1))` |
| `make_shape(M, N, K)` | `(M, N, K)` (tuple) |
| `make_stride(S0, S1, S2)` | `stride=(S0, S1, S2)` (keyword) |
| `cosize(layout)` | `cute.cosize(layout)` |
| `rank(layout)` | `cute.rank(layout)` |
| `depth(layout)` | `cute.depth(layout)` |

### Key Mental Model

**You already know this.** The algebra is identical — only the Python syntax differs:
- Tuples replace `make_shape`/`make_stride`
- Keyword arguments for clarity
- Same bijective mapping semantics

---

## Exercises

| Exercise | Topic | Difficulty | Job Relevance |
|----------|-------|------------|---------------|
| 01 | `make_layout` basics | [EASY] | Foundation for all tiled kernels |
| 02 | Shape and stride algebra | [EASY] | Understanding memory access patterns |
| 03 | Hierarchical layouts | [MEDIUM] | GQA, multi-head attention layouts |
| 04 | Stride-0 broadcast (GQA) | [HARD] | **Direct FlashAttention-3 optimization** |

---

## Why This Module Matters

### NVIDIA DL Software Engineer Interview
- Layout algebra is **question #1** in every CuTe interview loop
- Stride-0 broadcasting is the canonical GQA optimization

### FlashAttention / vLLM / TensorRT-LLM
- Every tiled kernel starts with layout definitions
- GQA uses stride-0 on the KV heads dimension to eliminate redundant loads

### Cerebras Wafer-Scale
- Layout partitioning maps directly to 2D mesh topology
- Hierarchical layouts = hierarchical core groups

---

## Profiling Focus

For Module 01, layouts are host-side data structures — no GPU profiling needed yet. Starting in Module 03 (TiledCopy), we'll use:

```bash
ncu --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum \
    --set full --target-processes all \
    python ex0X_...py
```

---

## Prerequisites

You already have these from CuTe C++ 3.x:
- Layout = bijective mapping (coords → linear index)
- `cosize` flattens to 1D
- Shape/stride duality
- Hierarchical layout composition

**New in 4.x Python:**
- Tuple syntax instead of `make_shape`
- `stride=` keyword argument

---

**Next:** Open `ex01_make_layout_FILL_IN.py` and answer the PREDICT questions before running code.
