# Project 14: Fused MLA (Multi-head Latent Attention)

## Objective

Implement Multi-head Latent Attention (MLA) - a memory-efficient attention variant. This project teaches:
- Latent space attention
- Query compression
- Multi-head latent decomposition
- Advanced attention optimization

## Theory

### MLA Overview

MLA reduces KV-cache memory by storing compressed latent representations:

```
Standard Attention:
  K, V: [batch, seq_len, num_heads * head_dim]

MLA:
  K_latent, V_latent: [batch, seq_len, num_latents * latent_dim]
  where num_latents << num_heads
```

### MLA Computation

```
1. Project Q to latent space: Q_latent = Q × W_Q_latent
2. Attention in latent space: A = softmax(Q_latent × K_latent^T)
3. Project back: O = A × V_latent × W_out
```

## Your Task

Implement MLA with:
- Latent space projection
- Compressed KV attention
- Output projection

---

**Ready for advanced attention? Open `mla.cu`!**
