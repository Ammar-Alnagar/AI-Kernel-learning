# Module 02: Tensors — From Layouts to Data

## What This Module Teaches

Tensors combine a layout with actual data (via pointers). You will learn to create tensors in global memory, slice them by head or sequence position, and use `local_tile` to iterate over K/V blocks — the exact pattern used in FlashAttention-2.

## Why Tensors Matter for LLM Inference

**Job mapping:** NVIDIA Deep Learning Software Engineer (Inference) — "Implement FlashAttention-2 with CuTe tensor operations."

In FlashAttention-2, you:
1. Create Q, K, V tensors from device pointers: `make_tensor(make_gmem_ptr(...), layout)`
2. Slice by head: `Q(_, _, head_idx, _)` for per-head processing
3. Iterate over K/V blocks with `local_tile`: process one tile at a time to fit in SRAM
4. Partition threads with `local_partition` for warp-level operations

## Prerequisites

- Module 01: Layouts (make_layout, logical_divide, stride)
- CUDA memory management (cudaMalloc, cudaMemcpy)
- Pointer arithmetic basics

## Exercises

| Exercise | Concept | LLM Pattern |
|----------|---------|-------------|
| ex01_tensor_creation.cu | `make_tensor`, `make_gmem_ptr`, `make_smem_ptr` | Q/K/V tensor construction |
| ex02_slicing_views.cu | Tensor slicing with `_` (underscore) | Per-head Q/K/V slicing |
| ex03_local_tile.cu | `local_tile` for block iteration | FlashAttention-2 K/V tile loop |
| ex04_local_partition.cu | `local_partition` for thread mapping | Warp-level fragment distribution |

## Exit Criteria

Before moving to Module 03, you must be able to:

1. Create a tensor from a device pointer and layout: `make_tensor(ptr, layout)`
2. Slice a 4D tensor to get a single head: `tensor(_, seq_dim, head_idx, _)`
3. Use `local_tile` to iterate over sequence blocks in a loop
4. Explain the difference between `local_tile` (block iteration) and `local_partition` (thread distribution)

## Common Mistakes

1. **Confusing tensor views with copies:** Slicing with `_` creates a view — no data is copied. The sliced tensor shares the same underlying memory.

2. **Forgetting to pass the layout to make_tensor:** `make_tensor` needs both a pointer AND a layout. The layout determines how indices map to memory offsets.

3. **Using local_tile incorrectly:** `local_tile(tensor, tile_shape, start_coord)` returns a view of ONE tile. You must loop over tile indices to process all tiles.

---

Next: **ex01_tensor_creation.cu** — your first CuTe tensor.
