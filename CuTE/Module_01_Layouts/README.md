# Module 01: Layouts — The Foundation of CuTe Tiling

## What This Module Teaches

Layouts are CuTe's core abstraction for describing tensor memory organization. A layout maps logical indices (i, j, k...) to physical memory offsets. You will learn to construct layouts for attention tensors, tile them with `logical_divide`, and express GQA patterns with stride-0 broadcast.

## Why Layouts Matter for LLM Inference

**Job mapping:** NVIDIA Deep Learning Software Engineer (Inference) — "Design and implement custom CUDA kernels using CuTe layout algebra for FlashAttention variants."

In FlashAttention-2, you tile the Q tensor into blocks of shape `[batch, heads, Br, head_dim]` where `Br` is the row tile (e.g., 64). The layout algebra lets you express this tiling as `logical_divide(shape, tile_shape)` and iterate over tiles without manual index arithmetic.

GQA (Grouped Query Attention) uses stride-0 broadcast: multiple query heads share the same K/V head. CuTe expresses this with a layout where one dimension has stride 0.

## Prerequisites

- CUDA pointer basics (`cudaMalloc`, `cudaMemcpy`, kernel launches)
- C++ templates and `constexpr`
- Row-major vs. column-major memory ordering

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_basic_layouts.cu | `make_layout`, `shape`, `stride`, `print_layout` | Row-major Q/K/V tensor layout |
| ex02_tiling_with_logical_divide.cu | `logical_divide` for tiling | FlashAttention-2 block tiling |
| ex03_attention_tensor_layouts.cu | Multi-dimensional layouts `[batch, heads, seqlen, head_dim]` | Full attention tensor shape |
| ex04_gqa_stride_zero.cu | Stride-0 broadcast layout | GQA shared K/V heads |

## Exit Criteria

Before moving to Module 02, you must be able to:

1. Construct a row-major layout for `[seqlen, head_dim]` and explain why stride is `(head_dim, 1)`
2. Use `logical_divide` to tile a `[128, 64]` layout into 4×4 tiles of shape `[32, 16]`
3. Write the layout for GQA where 8 query heads share 2 K/V heads (stride-0 on the head dimension for K/V)
4. Read a `print_layout` output and trace how index `(i, j)` maps to offset

## Common Mistakes

1. **Confusing shape and stride:** Shape is the logical dimensions. Stride is how many elements to skip in memory for each dimension increment. Row-major `[M, N]` has stride `(N, 1)`.

2. **Forgetting that `logical_divide` returns a composed layout:** The result has two levels — the original "big" shape and the tile shape. You access elements with a tuple of tuples: `layout(make_coord(i, j), make_coord(ii, jj))`.

3. **Assuming column-major is rare:** Tensor Core MMA uses column-major for some operands. Always check the CUTLASS documentation for the expected layout of each GEMM operand.

---

Next: **ex01_basic_layouts.cu** — your first CuTe layout.
