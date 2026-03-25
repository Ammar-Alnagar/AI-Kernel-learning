# Module 05: Matrix Operations

## Goal

Build intuition for GEMM tiling and Tensor Core style computation.

- tiled matmul decomposition
- accumulator tiles
- K-loop blocking
- MMA (conceptual) vs scalar FMA

## Exercises

1. `matmul_naive`: baseline triple-loop GEMM.
2. `matmul_tiled`: block GEMM with configurable tile sizes.
3. `mma_tile_update`: update one accumulator tile from A/B tiles.
