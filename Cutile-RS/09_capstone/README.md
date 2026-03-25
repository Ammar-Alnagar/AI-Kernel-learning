# Module 09: Capstone - Fused Multi-Head Attention

## Goal

Assemble a fused attention pipeline from previous modules.

- QK^T score computation
- scale + causal mask
- row-wise softmax
- weighted sum with V

## Exercises

1. `attention_scores`
2. `causal_mask_inplace`
3. `row_softmax`
4. `fused_attention`

This capstone uses NumPy to validate math. In CuTile kernels, each stage should be tiled and fused to reduce global-memory traffic.
