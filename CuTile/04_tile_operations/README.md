# Module 04: Tile Operations - Arithmetic, Broadcasting, and Shape Ops

## Learning Objectives

By the end of this module, you will:
- Apply element-wise arithmetic on tiles
- Use scalar and vector broadcasting
- Reshape tiles for layout-aware operations
- Transpose 2D tiles for memory-friendly access

## Concepts

Tile operations are compile-time shaped operations. This gives the compiler strong information for optimization.

Important patterns:
1. Fused element-wise expressions
2. Broadcasted scalar transforms
3. Layout conversion (`reshape`, `transpose`) before compute

## Exercises

1. Affine transform: `y = x * alpha + beta`
2. Broadcast add with a scalar
3. 2D transpose kernel
4. Normalize each tile with per-tile mean

## Run

```bash
python test.py
```
