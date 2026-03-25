# Module 06: Advanced Tiling - 2D/3D Grids and Swizzling

## Learning Objectives

By the end of this module, you will:
- Use 2D and 3D launch grids effectively
- Map block IDs to matrix and batch coordinates
- Apply simple swizzling for locality-aware mapping
- Compare coalesced vs non-coalesced access patterns

## Concepts

Advanced tiling is about mapping work to hardware efficiently.

Key ideas:
1. Multi-dimensional grid indexing (`ct.bid(0/1/2)`)
2. Logical-to-physical tile remapping (swizzle)
3. Batch-major and row-major access tradeoffs

## Exercises

1. 2D map kernel
2. 3D batched transform kernel
3. XOR swizzle for tile IDs
4. Coalesced memory path baseline

## Run

```bash
python test.py
```
