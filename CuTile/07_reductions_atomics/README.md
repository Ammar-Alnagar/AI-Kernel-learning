# Module 07: Reductions and Atomics

## Learning Objectives

By the end of this module, you will:
- Use `ct.sum()` and `ct.max()` on tiles
- Implement block-wise reductions
- Understand when atomics are required
- Build simple histogram-style accumulation patterns

## Concepts

Reductions combine many values into fewer values.

Common operations:
1. Sum reduction
2. Max reduction
3. Atomic-style accumulation patterns (or two-pass fallback)

Atomics are needed when multiple blocks may update the same location.

## Exercises

1. Block sum kernel
2. Block max kernel
3. Per-block partial accumulation kernel
4. Host-level reduction verification

## Run

```bash
python test.py
```
