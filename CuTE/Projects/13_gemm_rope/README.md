# Project 13: Fused GEMM + RoPE (Rotary Position Embedding)

## Objective

Implement GEMM with fused Rotary Position Embedding (RoPE). This project teaches:
- RoPE for positional encoding in transformers
- Fused kernel operations
- Complex multiplication for rotation
- In-place transformation

## Theory

### RoPE Formula

RoPE applies rotation to Q and K vectors based on position:

```
RoPE(x, pos) = x × exp(i × pos × θ)

where θ = 1 / (10000^(2d/d_model))
```

In real arithmetic (pairs of elements):
```
[q_2i, q_2i+1] → [q_2i*cos(mθ) - q_2i+1*sin(mθ),
                   q_2i*sin(mθ) + q_2i+1*cos(mθ)]
```

### Fused GEMM + RoPE

Instead of separate kernels:
```
Q_fp16 = GEMM(Q_input, W_Q)  # Write to HBM
Q_rope = RoPE(Q_fp16)         # Read/write HBM

# Fused:
Q_rope = RoPE(GEMM(Q_input, W_Q))  # Register only
```

## Your Task

Implement fused GEMM + RoPE:
- Compute GEMM output in registers
- Apply RoPE rotation before storing
- Handle even/odd element pairs

---

**Ready to rotate? Open `gemm_rope.cu`!**
