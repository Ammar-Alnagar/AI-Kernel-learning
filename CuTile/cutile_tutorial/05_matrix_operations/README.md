# Module 05: Matrix Operations - matmul, mma, Tensor Cores

## Learning Objectives

By the end of this module, you will:
- Build a tiled matrix multiplication kernel
- Use `ct.matmul()` for tile-level GEMM
- Understand `ct.mma()` as Tensor Core primitive style
- Accumulate partial products across K tiles

## Concepts

For C = A @ B:
- A shape: `(M, K)`
- B shape: `(K, N)`
- C shape: `(M, N)`

Tiled strategy:
1. Each block computes one `(tile_m, tile_n)` output tile
2. Loop over K in chunks of `tile_k`
3. Accumulate into an output tile accumulator

## Exercises

1. Single-tile matmul using `ct.matmul`
2. K-loop accumulation kernel
3. Tensor-Core style kernel with `ct.mma`
4. Host launch helper and correctness check

## Run

```bash
python test.py
```
