# CuTe DSL Learning Curriculum — CUTLASS 4.x Python API

## For: Ammar — Senior GPU Kernel Engineer (CuTe C++ 3.x → CuTe DSL 4.x)

This curriculum translates your deep CuTe C++ expertise into the modern Python CuteDSL API. You already understand layouts, `logical_divide`, `local_tile`, `TiledCopy`, `TiledMMA`, swizzle, and pipelines. This is **syntax translation**, not concept re-learning.

---

## 📁 Directory Structure

```
cute_dsl/
├── README.md                    # This file — learning path overview
├── setup.py                     # Environment validation + GPU detection
│
├── module_01_layouts/           # Layout algebra in Python
│   ├── README.md
│   ├── ex01_make_layout_FILL_IN.py
│   ├── ex01_make_layout_SOLUTION.py
│   ├── ex02_shape_stride_FILL_IN.py
│   ├── ex02_shape_stride_SOLUTION.py
│   ├── ex03_hierarchical_layouts_FILL_IN.py
│   ├── ex03_hierarchical_layouts_SOLUTION.py
│   └── ex04_gqa_stride0_broadcast_FILL_IN.py
│       └── ex04_gqa_stride0_broadcast_SOLUTION.py
│
├── module_02_tensors/           # Memory tensors: gmem, smem, rmem
│   ├── README.md
│   ├── ex01_gmem_tensor_FILL_IN.py
│   ├── ex01_gmem_tensor_SOLUTION.py
│   ├── ex02_smem_tensor_FILL_IN.py
│   ├── ex02_smem_tensor_SOLUTION.py
│   ├── ex03_rmem_tensor_FILL_IN.py
│   ├── ex03_rmem_tensor_SOLUTION.py
│   ├── ex04_slicing_views_FILL_IN.py
│   ├── ex04_slicing_views_SOLUTION.py
│   ├── ex05_local_tile_FILL_IN.py
│   └── ex05_local_tile_SOLUTION.py
│
├── module_03_tiled_copy/        # Data movement: TiledCopy, TMA
│   ├── README.md
│   ├── ex01_copy_atom_FILL_IN.py
│   ├── ex01_copy_atom_SOLUTION.py
│   ├── ex02_make_tiled_copy_tv_FILL_IN.py
│   ├── ex02_make_tiled_copy_tv_SOLUTION.py
│   ├── ex03_vectorized_gmem_to_smem_FILL_IN.py
│   ├── ex03_vectorized_gmem_to_smem_SOLUTION.py
│   └── ex04_tma_copy_hopper_FILL_IN.py
│       └── ex04_tma_copy_hopper_SOLUTION.py
│
├── module_04_tiled_mma/         # Compute: TiledMMA, GEMM mainloop
│   ├── README.md
│   ├── ex01_mma_atom_FILL_IN.py
│   ├── ex01_mma_atom_SOLUTION.py
│   ├── ex02_tiled_mma_setup_FILL_IN.py
│   ├── ex02_tiled_mma_setup_SOLUTION.py
│   ├── ex03_gemm_mainloop_FILL_IN.py
│   ├── ex03_gemm_mainloop_SOLUTION.py
│   └── ex04_mixed_precision_FILL_IN.py
│       └── ex04_mixed_precision_SOLUTION.py
│
├── module_05_swizzle/           # Shared memory banking, swizzle layouts
│   ├── README.md
│   ├── ex01_bank_conflict_visualizer_FILL_IN.py
│   ├── ex01_bank_conflict_visualizer_SOLUTION.py
│   ├── ex02_swizzle_smem_layout_FILL_IN.py
│   ├── ex02_swizzle_smem_layout_SOLUTION.py
│   └── ex03_verify_with_ncu_FILL_IN.py
│       └── ex03_verify_with_ncu_SOLUTION.py
│
├── module_06_pipeline/          # Double-buffer, async, warp-specialized
│   ├── README.md
│   ├── ex01_double_buffer_FILL_IN.py
│   ├── ex01_double_buffer_SOLUTION.py
│   ├── ex02_async_copy_overlap_FILL_IN.py
│   ├── ex02_async_copy_overlap_SOLUTION.py
│   └── ex03_warp_specialized_pipeline_FILL_IN.py
│       └── ex03_warp_specialized_pipeline_SOLUTION.py
│
├── module_07_predication/       # Predicated copies, irregular shapes
│   ├── README.md
│   ├── ex01_predicated_copy_FILL_IN.py
│   ├── ex01_predicated_copy_SOLUTION.py
│   └── ex02_irregular_tile_gemm_FILL_IN.py
│       └── ex02_irregular_tile_gemm_SOLUTION.py
│
├── module_08_mma_atom_internals/ # Fragment layout, ownership
│   ├── README.md
│   ├── ex01_tv_layout_inspection_FILL_IN.py
│   ├── ex01_tv_layout_inspection_SOLUTION.py
│   └── ex02_fragment_ownership_FILL_IN.py
│       └── ex02_fragment_ownership_SOLUTION.py
│
└── projects/                    # Capstone production kernels
    ├── 01_tiled_gemm/           # Target: >75% roofline SM80
    ├── 02_online_softmax/       # Target: >85% BW utilization
    ├── 03_multihead_attention/  # Unfused → fused progression
    ├── 04_flash_attention_2/    # Dao et al. tiled FA2
    ├── 05_flash_attention_3/    # Shah et al. warp-specialized FA3
    ├── 06_fused_attention_variants/  # GQA, MLA, sliding window
    ├── 07_quantized_gemm/       # INT8, FP8 (E4M3/E5M2)
    └── 08_benchmarks_master/    # Roofline charts, C++ vs DSL comparison
```

---

## 🧠 C++ ↔ DSL Concept Bridge Table

| Concept              | CuTe C++ 3.x                                    | CuteDSL 4.x Python                                |
|----------------------|-------------------------------------------------|---------------------------------------------------|
| **Layout creation**  | `make_layout(make_shape(M,N), make_stride(N, 1))` | `cute.make_layout((M, N), stride=(N, 1))`         |
| **Shape/Stride**     | `make_shape(M, N, K)`, `make_stride(S0, S1, S2)` | Tuples: `(M, N, K)`, `(S0, S1, S2)`               |
| **Cosize**           | `cosize(layout)`                                | `cute.cosize(layout)` — identical                 |
| **Local tile**       | `local_tile(tensor, tile_shape, tile_coord)`    | `cute.local_tile(tensor, tile_shape, tile_coord)` |
| **Local partition**  | `local_partition(tensor, thr_layout, tid)`      | `cute.local_partition(tensor, thr_layout, tid)`   |
| **TiledCopy setup**  | `make_tiled_copy(Copy_Atom{}, thr_layout, val_layout)` | `cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)` |
| **Copy execution**   | `copy(tiled_copy, src, dst)`                    | `cute.copy(tiled_copy, src, dst)`                 |
| **Predicated copy**  | Manual predicate logic                          | `cute.copy(atom, src, dst, pred=pred_tensor)`     |
| **Register fragment**| `thr_mma.partition_fragment(...)`               | `cute.make_rmem_tensor_like(tensor)`              |
| **Clear accum**      | `clear(accum)`                                  | `cute.clear(accum)`                               |
| **TiledMMA**         | `gemm(tiled_mma, D, A, B, C)`                   | `cute.gemm(tiled_mma, D, A, B, C)`                |
| **Swizzle**          | `composition(Swizzle<B,M,S>{}, layout)`         | `cute.composition(cute.Swizzle(B, M, S), layout)` |
| **JIT kernel**       | `__global__ void kernel(...)`                   | `@cutlass.jit` / `@cutlass.cute.kernel`           |
| **Static int**       | `Int<4>{}`                                      | `cutlass.Constexpr(4)` or just `4` (inferred)     |
| **Barrier sync**     | `fence_view_async_mbarrier()`                   | `pipeline.sync()` / `cute.fence()`                |

### Key Mental Model Shifts

1. **Tuples replace `make_shape`/`make_stride`**: Python's native tuples carry the same algebraic meaning. `(M, N)` is a shape; `(N, 1)` is a stride.

2. **`make_tiled_copy_tv` replaces `make_tiled_copy`**: The `_tv` suffix makes the **thread-value layout** explicit. This avoids implicit broadcast ambiguity in the older API.

3. **`rmem_tensor` is first-class**: In C++ 3.x, register fragments came from `partition_fragment`. In 4.x Python, `make_rmem_tensor` is the canonical way.

4. **Predication is a keyword argument**: `pred=pred_tensor` in `cute.copy()` — much cleaner than manual predicate layout algebra.

---

## 🗺️ Learning Path Diagram

```
                    ┌─────────────────────────────────────────────────┐
                    │  MODULE 01: Layouts (you are here)              │
                    │  • make_layout, shape, stride, cosize           │
                    │  • Hierarchical layouts (GQA stride-0)          │
                    └─────────────────────────────────────────────────┘
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │  MODULE 02: Tensors                             │
                    │  • gmem, smem, rmem tensors                     │
                    │  • local_tile, slicing views                    │
                    └─────────────────────────────────────────────────┘
                                         ↓
        ┌────────────────────────────────┴────────────────────────────────┐
        ↓                                                                 ↓
┌───────────────────┐                                         ┌─────────────────────┐
│ MODULE 03:        │                                         │ MODULE 04:          │
│ TiledCopy         │                                         │ TiledMMA            │
│ • Copy atoms      │                                         │ • MMA atoms         │
│ • make_tiled_copy │                                         │ • TiledMMA setup    │
│ • Vectorized load │                                         │ • GEMM mainloop     │
│ • TMA (SM90+)     │                                         │ • Mixed precision   │
└───────────────────┘                                         └─────────────────────┘
        ↓                                                                 ↓
        └────────────────────────────────┬────────────────────────────────┘
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │  MODULE 05: Swizzle                             │
                    │  • Bank conflict visualization                  │
                    │  • Swizzle layouts for SMEM                     │
                    │  • Nsight Compute verification                  │
                    └─────────────────────────────────────────────────┘
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │  MODULE 06: Pipelines                           │
                    │  • Double-buffer                                │
                    │  • Async copy overlap                           │
                    │  • Warp-specialized (DMA/MMA split)             │
                    └─────────────────────────────────────────────────┘
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │  MODULE 07: Predication                         │
                    │  • Predicated copies                            │
                    │  • Irregular tile GEMM                          │
                    └─────────────────────────────────────────────────┘
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │  MODULE 08: MMA Atom Internals                  │
                    │  • TV layout inspection                         │
                    │  • Fragment ownership                           │
                    └─────────────────────────────────────────────────┘
                                         ↓
                    ┌─────────────────────────────────────────────────┐
                    │  PROJECTS 01-08: Capstone Kernels               │
                    │  • Tiled GEMM (>75% roofline)                   │
                    │  • FlashAttention 2 & 3                         │
                    │  • GQA, MLA, FP8                                │
                    └─────────────────────────────────────────────────┘
```

---

## ⚙️ Setup Instructions

### 1. Install the Package

```bash
pip install nvidia-cutlass-dsl
```

### 2. Validate Your Environment

```bash
python setup.py
```

Expected output:
```
✓ nvidia-cutlass-dsl installed: v4.4.x
✓ GPU detected: NVIDIA H100 (SM90)
✓ CUDA toolkit: 12.x
✓ Roofline peak (FP16 Tensor): 989 TFLOPS
✓ Hello-world layout: Layout[(M=128, N=64), stride=(64, 1)]
```

### 3. Run Module 01 Exercises

```bash
cd module_01_layouts
python ex01_make_layout_FILL_IN.py   # Attempt first
python ex01_make_layout_SOLUTION.py  # Then verify
```

### 4. Profile with Nsight Compute

```bash
ncu --set full --target-processes all python ex01_make_layout_FILL_IN.py
```

---

## ✅ Prerequisite Check (You Already Have These)

Since you're a CuTe C++ 3.x expert, these concepts transfer **directly** — only the Python syntax is new:

| You Already Know                          | Python Syntax Translation                    |
|-------------------------------------------|----------------------------------------------|
| `make_layout(make_shape(M, N), ...)`      | `cute.make_layout((M, N), ...)`              |
| `make_stride(N, 1)` for row-major         | `stride=(N, 1)` keyword argument             |
| `cosize(layout)` flattens to 1D           | `cute.cosize(layout)` — identical            |
| `local_tile(tensor, tile_shape, coord)`   | `cute.local_tile(...)` — identical           |
| Thread/value layout partitioning          | Same algebra, tuple syntax                   |
| `Swizzle<B, M, S>` for bank avoidance     | `cute.Swizzle(B, M, S)` — constructor call   |
| `Copy_Atom<...>` for TiledCopy            | `cute.Copy_atom(...)` or string identifiers  |
| `MMA_Atom<...>` for TiledMMA              | `cute.MMA_atom(...)` or string identifiers   |
| Pipeline async protocols                  | `cutlass.pipeline.PipelineAsync`             |
| Barrier/warp-group synchronization        | `pipeline.sync()` / `cute.fence()`           |

**What's genuinely new in 4.x Python:**
- `make_tiled_copy_tv` (replaces `make_tiled_copy`)
- `make_rmem_tensor` (replaces `partition_fragment` for registers)
- `pred=` keyword in `cute.copy()`
- `@cutlass.jit` decorator instead of `__global__` kernel syntax

---

## 🎯 Why This Module Matters (Job Relevance)

### Modular Senior AI Kernel Engineer
- **Layout algebra** is the foundation of every tiled kernel — GEMM, attention, conv
- **Stride-0 broadcasting** (GQA) eliminates redundant memory loads — critical for multi-query inference

### Cerebras Performance Engineer — Inference
- **Wafer-scale kernels** use identical layout partitioning across thousands of cores
- **Hierarchical layouts** map directly to Cerebras's 2D mesh topology

### NVIDIA Senior DL Software Engineer, Inference
- **Every interview question** on CuTe starts with layout algebra
- **FlashAttention, vLLM, TensorRT-LLM** all use these exact patterns
- **GQA stride-0** is the canonical optimization for Llama-2/3 multi-query attention

---

## 📚 Required Reading (Before Module 01)

1. **CUTLASS 4.x Python API Docs**: https://nvidia.github.io/cutlass-dsl/
2. **CuTe DSL Examples**: https://github.com/NVIDIA/cutlass/tree/main/examples/python/CuTeDSL
3. **Layout Algebra Primer**: https://github.com/NVIDIA/cutlass/blob/main/media/docs/cute_layout.md

---

## 🚀 Next Steps

1. Run `python setup.py` to validate your environment
2. Open `module_01_layouts/ex01_make_layout_FILL_IN.py`
3. Answer the **PREDICT** questions before running any code
4. Attempt the fill-in zones [EASY] → [MEDIUM] → [HARD]
5. Compare with `_SOLUTION.py` and profile with `ncu`

**When you complete Module 01**, move to `module_02_tensors/`.

---

## 📊 Benchmark Output Format

Every project will output tables like this:

```
┌──────────────────────┬──────────────┬──────────────┬──────────────┐
│ Kernel               │ TFLOPS       │ % Roofline   │ vs Baseline  │
├──────────────────────┼──────────────┼──────────────┼──────────────┤
│ Naive GEMM           │ 45.2         │ 14.5%        │ 1.0×         │
│ Tiled GEMM (DSL)     │ 198.7        │ 63.7%        │ 4.4×         │
│ CUTLASS C++ ref      │ 241.3        │ 77.3%        │ 5.3×         │
└──────────────────────┴──────────────┴──────────────┴──────────────┘
```

---

**Let's build production kernels that hit roofline.** 🚀
