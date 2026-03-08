# Module 05 — Swizzle: Shared Memory Banking

## Concept Overview

Swizzle layouts reorder memory addresses to avoid bank conflicts in shared memory. When multiple threads access SMEM simultaneously, conflicting accesses to the same memory bank serialize — destroying throughput. Swizzling XORs address bits to distribute accesses across banks.

### C++ → DSL Translation

| C++ 3.x | Python 4.x |
|---------|------------|
| `composition(Swizzle<B,M,S>{}, layout)` | `cute.composition(cute.Swizzle(B, M, S), layout)` |
| `swizzle_bits<6, 3, 3>` | `cute.Swizzle(6, 3, 3)` |
| Bank conflict analysis | `cute.swizzle_bank_conflicts(layout)` |

### Swizzle Parameters

- **B**: XOR base bit position
- **M**: XOR mask bit position  
- **S**: Shift amount (which bits to XOR)

Common swizzle for 128-byte SMEM lines: `Swizzle(6, 3, 3)`

---

## Exercises

| Exercise | Topic | Difficulty | Job Relevance |
|----------|-------|------------|---------------|
| 01 | Bank conflict visualization | [MEDIUM] | Understanding SMEM bottlenecks |
| 02 | Swizzle SMEM layout | [HARD] | **Critical for GEMM/attention perf** |
| 03 | Verify with Nsight Compute | [HARD] | Production profiling workflow |

---

## Why This Module Matters

### NVIDIA DL Software Engineer Interview
- Bank conflict avoidance is a **standard interview question**
- Swizzling demonstrates deep understanding of GPU memory architecture

### FlashAttention / vLLM / TensorRT-LLM
- All high-performance kernels use swizzled SMEM layouts
- Without swizzling, GEMM achieves <50% of peak SMEM bandwidth

---

**Next:** Open `ex01_bank_conflict_visualizer_FILL_IN.py`
