# Causal Masking

## What This Is

Causal masking prevents attention from attending to future tokens during training and autoregressive generation. It enforces the constraint that token $i$ can only attend to tokens $j \leq i$. This is implemented as a triangular mask applied to the attention scores before softmax.

**The mask:**
$$M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{if } j > i \end{cases}$$

**Masked attention:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

## Why A Kernel Engineer Needs This

**This mask is applied inside your FlashAttention kernel as a comparison operation during the tile loop.** In CuTe, you implement this as a predicate: `if (col_idx > row_idx) score = -INFINITY`. The mask structure differs between prefill (full triangular) and decode (identity or no mask), and your kernel must handle both cases efficiently.

**Interview relevance:** Cerebras and NVIDIA interviewers ask about the difference between prefill and decode masking. Understanding this distinction is critical for implementing efficient inference kernels.

## The Math

### The Triangular Mask Matrix

For sequence length $S = 4$, the causal mask matrix is:

$$M = \begin{bmatrix}
0 & -\infty & -\infty & -\infty \\
0 & 0 & -\infty & -\infty \\
0 & 0 & 0 & -\infty \\
0 & 0 & 0 & 0
\end{bmatrix}$$

**Interpretation:**
- Row 0 (token 0): can only attend to token 0
- Row 1 (token 1): can attend to tokens 0, 1
- Row 2 (token 2): can attend to tokens 0, 1, 2
- Row 3 (token 3): can attend to tokens 0, 1, 2, 3

**Why $-\infty$?** Because $\exp(-\infty) = 0$, so after softmax, masked positions have zero probability:
$$\text{softmax}(x)_i = \frac{\exp(x_i)}{\sum_j \exp(x_j)} \quad \Rightarrow \quad \text{if } x_i = -\infty, \text{ then } \exp(x_i) = 0$$

### Implementation via Comparison

The mask can be computed on-the-fly without materializing the full matrix:

$$M_{i,j} = \begin{cases} 0 & \text{if } j \leq i \\ -\infty & \text{otherwise} \end{cases}$$

In code:
```python
# i = row index (query position), j = col index (key position)
mask = (j > i) ? -INFINITY : 0.0f
```

**This is a simple comparison — no memory access needed.** In CUDA, this is a single `if` statement or a predicated store.

### Causal Attention Formula

$$\text{CausalAttention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V$$

**Step-by-step for position $i$:**

1. Compute raw scores: $s_{i,j} = \frac{Q_i \cdot K_j}{\sqrt{d_k}}$ for all $j \in [0, S)$

2. Apply mask: $\hat{s}_{i,j} = s_{i,j} + M_{i,j} = \begin{cases} s_{i,j} & j \leq i \\ -\infty & j > i \end{cases}$

3. Softmax: $p_{i,j} = \frac{\exp(\hat{s}_{i,j})}{\sum_{k=0}^{S} \exp(\hat{s}_{i,k})}$

4. Weighted sum: $o_i = \sum_{j=0}^{S} p_{i,j} V_j$

**Key observation:** After masking, the softmax denominator only sums over $j \leq i$:
$$\sum_{k=0}^{S} \exp(\hat{s}_{i,k}) = \sum_{k=0}^{i} \exp(s_{i,k})$$

This matters for online softmax (Module 05).

## Shapes and Sizes

| Operation | Input shapes | Output shape |
|-----------|--------------|--------------|
| Causal mask generation | $S_q, S_k$ (scalars) | $[S_q, S_k]$ (computed on-the-fly) |
| Masked scores | $S: [B,H,S_q,S_k]$, $M: [S_q,S_k]$ | $[B,H,S_q,S_k]$ |

**Critical:** The mask is NOT stored as a tensor. It is computed as a comparison during the score computation. Memory: $O(1)$.

## The Kernel Implication

### Prefill vs Decode Masking

**Prefill (training or prompt processing):**
- Process all $S$ tokens in parallel
- Full triangular mask required
- Token $i$ attends to tokens $0, 1, \ldots, i$

```cuda
// Prefill: full causal mask
__device__ float apply_causal_mask(float score, int q_pos, int k_pos) {
    return (k_pos > q_pos) ? -INFINITY : score;
}
```

**Decode (autoregressive generation):**
- Generate one token at a time
- Current token $S$ attends to all previous tokens $0, 1, \ldots, S-1$
- No triangular mask needed — the "future" is empty!

```cuda
// Decode: no mask needed (or identity mask)
// Token S attends to 0, 1, ..., S-1 (all are "past")
__device__ float apply_decode_mask(float score, int q_pos, int k_pos) {
    // q_pos == S (current), k_pos in [0, S) (all past)
    // No masking needed — all keys are in the past
    return score;
}
```

**Why the difference?**

In decode, you generate token $S$ given tokens $0, \ldots, S-1$. There are no "future" tokens to mask — all existing tokens are in the past. The causal constraint is automatically satisfied.

**This is why KV cache works:** In decode, you reuse cached $K, V$ for positions $0, \ldots, S-1$ and only compute new $K, V$ for position $S$. No masking overhead.

### FlashAttention Tile Loop with Causal Mask

```cuda
// Simplified FlashAttention with causal masking
for (int tile_j = 0; tile_j < num_tiles; ++tile_j) {
    int k_start = tile_j * BLOCK_SIZE;
    int k_end = k_start + BLOCK_SIZE;
    
    // Load K, V tiles into SRAM
    load_k_tile(K + k_start, k_tile);
    load_v_tile(V + k_start, v_tile);
    
    // Compute QK^T for this tile
    for (int i = 0; i < BLOCK_SIZE; ++i) {
        int q_pos = tile_i * BLOCK_SIZE + i;
        for (int j = 0; j < BLOCK_SIZE; ++j) {
            int k_pos = k_start + j;
            
            // CAUSAL MASK: skip if attending to future
            if (k_pos > q_pos) {
                scores[i][j] = -INFINITY;
            } else {
                scores[i][j] = dot(q_tile[i], k_tile[j]) / sqrt_dk;
            }
        }
    }
    
    // Online softmax update (Module 05)
    update_softmax_max_sum(scores);
}
```

**Key point:** The causal mask is a simple `if (k_pos > q_pos)` check inside the tile loop. No extra memory, no extra kernel.

## Numbers That Matter

| Scenario | S | Mask type | Mask memory | Mask compute |
|----------|---|-----------|-------------|--------------|
| Prefill | 4096 | Triangular | $O(1)$ (computed) | 1 comparison per score |
| Prefill | 8192 | Triangular | $O(1)$ (computed) | 1 comparison per score |
| Decode | 128 | None | $O(1)$ | 0 comparisons |
| Decode | 512 | None | $O(1)$ | 0 comparisons |

**Note:** The mask is computed on-the-fly. Memory is $O(1)$ regardless of sequence length.

## Common Interview Questions

**Q1: Why does decode not need causal masking?**

<details>
<summary>Answer</summary>

In autoregressive decode, you generate one token at a time. When generating token $S$, the only existing tokens are $0, 1, \ldots, S-1$ — all of which are in the past. There are no "future" tokens to mask.

The causal constraint is: "token $i$ can only attend to tokens $j \leq i$". In decode, token $S$ attends to tokens $0, \ldots, S-1$, and all satisfy $j < S$. The constraint is automatically satisfied.

This is why decode with KV cache is simpler than prefill: no masking logic needed.
</details>

**Q2: How do you implement causal masking in a FlashAttention tile loop without materializing the full mask matrix?**

<details>
<summary>Answer</summary>

Compute the mask on-the-fly as a comparison:

```cuda
for each score at (q_pos, k_pos):
    if (k_pos > q_pos):
        score = -INFINITY
    else:
        score = dot(Q[q_pos], K[k_pos]) / sqrt(dk)
```

This requires:
- Knowing the global position of each query and key in the tile
- A single comparison per score
- No extra memory

In CuTe, you use the layout to compute global positions from tile-local indices.
</details>

**Q3: What is the memory complexity of causal masking? Why?**

<details>
<summary>Answer</summary>

$O(1)$ — constant memory, independent of sequence length.

The mask is NOT stored as a tensor. It is computed as a comparison (`k_pos > q_pos`) during score computation. Each thread computes one score and applies the mask in-place.

If you materialized the full mask matrix, it would be $O(S^2)$, but no implementation does this.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01.1 (scaled dot-product attention) — you need to understand the base attention formula.

**What this unlocks:**
- Module 02 (KV Cache): Decode masking is trivial, which is why KV cache enables efficient autoregressive generation.
- Module 05 (FlashAttention): The tile loop must handle causal masking correctly, especially near the diagonal.
- Module 08 (Speculative Decoding): Tree attention uses a different mask structure (non-triangular).

**Next:** `03_multi_head_attention.md` — QKV projections and the complete multi-head attention FLOP count.
