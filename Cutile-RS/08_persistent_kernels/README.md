# Module 08: Persistent Kernels

## Goal

Understand persistent scheduling where each block processes multiple tiles.

- tile count and stride loops
- persistent work assignment
- occupancy-aware thinking

## Exercises

1. `persistent_tile_ids`: list tile ids handled by one block.
2. `persistent_vector_add`: process a vector by jumping tile stride each loop.
