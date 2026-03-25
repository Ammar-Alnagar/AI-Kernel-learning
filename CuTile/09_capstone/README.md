# Module 09: Capstone - Fused Multi-Head Attention (FMHA)

## Learning Objectives

By the end of this module, you will:
- Compose previously learned CuTile patterns into one fused workflow
- Implement core FMHA steps with tiled kernels
- Work with batched 4D tensor layouts `(B, H, S, D)`
- Validate numerical behavior against PyTorch reference attention

## Capstone Scope

This capstone implements a tutorial-sized FMHA pipeline:
1. Compute scores: `Q @ K^T`
2. Apply scale and row-wise normalization
3. Multiply by `V` to produce attention output

Focus is educational clarity, not production-level flash attention performance.

## Tensor Shapes

- `Q`: `(B, H, S, D)`
- `K`: `(B, H, S, D)`
- `V`: `(B, H, S, D)`
- `Scores`: `(B, H, S, S)`
- `Output`: `(B, H, S, D)`

## Exercises

1. Fused score kernel for one tile
2. Row-wise normalization helper
3. Fused output kernel (`P @ V`)
4. End-to-end host launcher

## Run

```bash
python test.py
```
