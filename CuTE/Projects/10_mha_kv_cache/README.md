# Project 10: Multi-Head Attention with KV-Cache

## Objective

Implement multi-head attention with KV-cache for efficient autoregressive generation. This project teaches:
- Multi-head attention mechanism
- KV-cache for incremental decoding
- Position-based indexing
- Memory-efficient attention

## Theory

### Multi-Head Attention

```
Attention(Q, K, V) = softmax(QK^T / √d) × V
MHA = Concat(head_1, ..., head_h) × W_O
head_i = Attention(Q×W_Qᵢ, K×W_Kᵢ, V×W_Vᵢ)
```

### KV-Cache

For autoregressive generation, cache K and V to avoid recomputation:

```
Step t:
Q_t: [batch, 1, d]      (current token)
K_cache: [batch, t, d]  (all previous keys)
V_cache: [batch, t, d]  (all previous values)
```

## Your Task

Implement MHA with:
- Multiple attention heads
- KV-cache loading
- Incremental attention (one token at a time)

---

**Ready to accelerate LLMs? Open `mha_kv_cache.cu`!**
