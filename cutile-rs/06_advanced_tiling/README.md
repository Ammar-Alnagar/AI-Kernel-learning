# Module 06: Advanced Tiling

## Goal

Move from 1D launch mapping to multi-dimensional tile scheduling.

- 2D grid decomposition
- block to tile coordinate mapping
- grouped ordering (swizzle-like behavior)

## Exercises

1. `linear_pid_to_2d`: convert linear block id to `(pid_m, pid_n)`.
2. `grouped_pid_to_2d`: grouped M-major traversal for cache reuse.
3. `tile_bounds`: produce row/col ranges for one tile.
