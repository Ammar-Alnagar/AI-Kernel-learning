# Module 02 — Tensors: GMEM, SMEM, RMEM, and Views

## Concept Overview

Tensors in CuTe DSL are **views** over memory with an associated layout. The memory space (global, shared, register) is explicit in the tensor creation API. This module covers the complete tensor hierarchy used in production kernels.

### C++ → DSL Translation

| C++ 3.x | Python 4.x |
|---------|------------|
| `make_tensor(ptr, layout)` | `cute.make_tensor(ptr, layout)` |
| `make_gmem_tensor(ptr, shape, stride)` | `cute.make_gmem_tensor(ptr, shape, stride)` |
| `make_smem_tensor(ptr, layout)` | `cute.make_smem_tensor(ptr, layout)` |
| `partition_fragment(...)` | `cute.make_rmem_tensor(shape, dtype)` |
| `local_tile(tensor, tile, coord)` | `cute.local_tile(tensor, tile, coord)` |
| `local_partition(tensor, thr, tid)` | `cute.local_partition(tensor, thr, tid)` |

### Memory Space Hierarchy

```
GMEM (global) ──→ SMEM (shared) ──→ RMEM (register)
     ↓                ↓                  ↓
  HBM/L2           On-chip           Thread-local
  ~1000 GB/s       ~19 TB/s          ~100+ TB/s
  High latency     Low latency       Zero latency
  All threads      CTA-wide          Per-thread
```

---

## Exercises

| Exercise | Topic | Difficulty | Job Relevance |
|----------|-------|------------|---------------|
| 01 | GMEM tensor creation | [EASY] | Foundation for all kernel I/O |
| 02 | SMEM tensor with shared memory pointer | [EASY] | Shared memory buffering |
| 03 | RMEM tensor (register fragments) | [MEDIUM] | MMA operand fragments |
| 04 | Slicing and views | [MEDIUM] | Zero-copy tensor views |
| 05 | `local_tile` for blocked access | [HARD] | **FlashAttention tiling core** |

---

## Why This Module Matters

### NVIDIA DL Software Engineer Interview
- Tensor creation and memory space selection is **fundamental** to kernel design questions
- Register pressure management (RMEM) is a key optimization discussion point

### FlashAttention / vLLM / TensorRT-LLM
- GMEM→SMEM→RMEM pipeline is the **data movement backbone** of every attention kernel
- `local_tile` is how FA2/FA3 partition the QKV sequences into blocks

### Cerebras Wafer-Scale
- Memory hierarchy is even more critical at wafer scale
- Explicit memory space control maps to Cerebras's software-defined memory

---

## Prerequisites

You already have from CuTe C++ 3.x:
- Tensor = pointer + layout
- Memory space semantics (gmem/smem/rmem)
- `local_tile` for blocked access
- View semantics (slicing is zero-copy)

**New in 4.x Python:**
- `make_rmem_tensor` replaces `partition_fragment`
- Cleaner slicing syntax with Python slices
- `from_dlpack` for PyTorch interop

---

**Next:** Open `ex01_gmem_tensor_FILL_IN.py`
