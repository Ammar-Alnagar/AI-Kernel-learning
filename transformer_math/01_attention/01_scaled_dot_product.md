# Scaled Dot-Product Attention

## What This Is

Attention is a weighted sum operation where the weights are computed from the similarity between queries and keys. Given a query $Q$ and a set of key-value pairs $(K, V)$, attention produces an output that is a weighted combination of the values, where the weight for each value is determined by how similar its corresponding key is to the query.

**The formula:**
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

This single formula is the core of every transformer layer. Everything in this directory exists to make this operation feasible at production scale.

## Why A Kernel Engineer Needs This

**This is the exact computation you implement inside your FlashAttention CuTe kernel.** Every optimization — tiling, online softmax, register sharing — must preserve this mathematical result. When you write your `gemm_QK` and `gemm_PV` functions in CuTe, you are computing $QK^T$ and then multiplying by $V$. The $\sqrt{d_k}$ scaling factor appears in your kernel as a compile-time constant or a single FMA instruction.

**Interview relevance:** NVIDIA interviewers will ask you to derive this formula and explain why each term exists. You must be able to write out the tensor shapes at every step.

## The Math

### Step 1: Input Shapes

We work with 4D tensors throughout. This is non-negotiable for multi-head attention.

$$Q \in \mathbb{R}^{B \times H \times S_q \times d_h}$$
$$K \in \mathbb{R}^{B \times H \times S_k \times d_h}$$
$$V \in \mathbb{R}^{B \times H \times S_k \times d_h}$$

Where:
- $B$ = batch size (number of sequences processed in parallel)
- $H$ = number of attention heads
- $S_q$ = query sequence length
- $S_k$ = key/value sequence length (equals $S_q$ in self-attention)
- $d_h$ = head dimension (also written as $d_k$ in some papers)

**Worked example (LLaMA-3 8B, prefill):**
- $B = 1$, $H = 32$, $S_q = S_k = 4096$, $d_h = 128$
- $Q$ shape: $(1, 32, 4096, 128)$
- $K$ shape: $(1, 32, 4096, 128)$
- $V$ shape: $(1, 32, 4096, 128)$

### Step 2: Query-Key Similarity ($QK^T$)

Compute the dot product between each query and each key. This produces a similarity score for every query-key pair.

$$S = QK^T$$

**Shape analysis:**
$$Q: [B, H, S_q, d_h]$$
$$K^T: [B, H, d_h, S_k] \quad \text{(transpose last two dimensions)}$$
$$S: [B, H, S_q, S_k]$$

**Worked example:**
- $Q$: $(1, 32, 4096, 128)$
- $K^T$: $(1, 32, 128, 4096)$
- $S = QK^T$: $(1, 32, 4096, 4096)$

**Critical observation:** The intermediate tensor $S$ has size $O(S^2)$. For $S = 4096$, this is $4096^2 = 16,777,216$ elements per head. With $H = 32$ heads and FP16 (2 bytes), this is:
$$32 \times 16,777,216 \times 2 \text{ bytes} = 1,073,741,824 \text{ bytes} \approx 1 \text{ GB}$$

**This is the memory wall.** Naive attention materializes this $O(S^2)$ tensor in HBM. FlashAttention's core insight is: never materialize $S$ — compute it tile by tile in SRAM.

### Step 3: Scaling by $\sqrt{d_k}$

$$\hat{S} = \frac{S}{\sqrt{d_k}} = \frac{QK^T}{\sqrt{d_k}}$$

**Why does $\sqrt{d_k}$ exist?**

Without scaling, the dot products $QK^T$ grow with $d_k$. Consider:
- Each element of $Q$ and $K$ has variance $\sigma^2$ (typically $\sigma^2 = 1/d_h$ after initialization)
- The dot product of two $d_h$-dimensional vectors has variance $d_h \cdot \sigma^2 \cdot \sigma^2 = d_h \cdot (1/d_h)^2 = 1/d_h$... wait, that's wrong.

Let me derive this correctly:

- Assume $Q_{ij} \sim \mathcal{N}(0, 1/d_h)$ and $K_{ij} \sim \mathcal{N}(0, 1/d_h)$ (Xavier initialization)
- The dot product $s_i = \sum_{j=1}^{d_h} Q_{ij} K_{ij}$ has variance:
  $$\text{Var}(s_i) = \sum_{j=1}^{d_h} \text{Var}(Q_{ij} K_{ij}) = \sum_{j=1}^{d_h} \frac{1}{d_h} \cdot \frac{1}{d_h} = d_h \cdot \frac{1}{d_h^2} = \frac{1}{d_h}$$

Wait, that gives variance $1/d_h$, which gets smaller with larger $d_h$. The issue is the opposite — let me reconsider with standard normal initialization:

- Assume $Q_{ij} \sim \mathcal{N}(0, 1)$ and $K_{ij} \sim \mathcal{N}(0, 1)$
- $\text{Var}(Q_{ij} K_{ij}) = E[Q_{ij}^2]E[K_{ij}^2] = 1 \cdot 1 = 1$ (independence)
- $\text{Var}(s_i) = \sum_{j=1}^{d_h} 1 = d_h$
- Standard deviation: $\sqrt{d_h}$

**So the dot products have standard deviation $\sqrt{d_h}$.** Without scaling, for $d_h = 128$, the scores have std dev $\approx 11.3$. After softmax, this pushes probabilities to 0 or 1 (saturation), causing gradient vanishing.

**The fix:** Divide by $\sqrt{d_h}$ to restore unit variance:
$$\text{Var}\left(\frac{s_i}{\sqrt{d_h}}\right) = \frac{1}{d_h} \text{Var}(s_i) = \frac{1}{d_h} \cdot d_h = 1$$

**This is why the scaling factor exists:** to prevent softmax saturation and gradient vanishing.

### Step 4: Softmax Normalization

$$P = \text{softmax}(\hat{S}) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

The softmax is applied along the last dimension (over keys). For each query position $i$ and head $h$:

$$P_{h,i,j} = \frac{\exp(\hat{S}_{h,i,j})}{\sum_{k=1}^{S_k} \exp(\hat{S}_{h,i,k})}$$

**Shape:** $P \in \mathbb{R}^{B \times H \times S_q \times S_k}$ (same as $S$)

**Properties:**
- Each row sums to 1: $\sum_{j=1}^{S_k} P_{h,i,j} = 1$
- All values in $[0, 1]$
- Interpreted as attention weights: "how much does query $i$ attend to key $j$?"

### Step 5: Weighted Sum of Values

$$O = PV$$

**Shape analysis:**
$$P: [B, H, S_q, S_k]$$
$$V: [B, H, S_k, d_h]$$
$$O: [B, H, S_q, d_h]$$

**Worked example:**
- $P$: $(1, 32, 4096, 4096)$
- $V$: $(1, 32, 4096, 128)$
- $O$: $(1, 32, 4096, 128)$

**Interpretation:** Each output token $O_{h,i,:}$ is a weighted average of all value vectors, where the weights come from how similar query $i$ is to each key.

### Step 6: Complete Formula with Shapes

$$\boxed{\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V}$$

$$[B, H, S_q, d_h] = \text{softmax}\left(\frac{[B,H,S_q,d_h] \times [B,H,S_k,d_h]^T}{\sqrt{d_k}}\right) \times [B, H, S_k, d_h]$$

$$[B, H, S_q, d_h] = \underbrace{\text{softmax}\left([B, H, S_q, S_k]\right)}_{[B, H, S_q, S_k]} \times [B, H, S_k, d_h]$$

## Shapes and Sizes

| Operation | Input shapes | Output shape | Memory (LLaMA-3 8B, S=4096, FP16) |
|-----------|--------------|--------------|-----------------------------------|
| $QK^T$ matmul | $Q: [B,H,S,d_h]$, $K^T: [B,H,d_h,S]$ | $[B,H,S,S]$ | 1 GB |
| Scale by $\sqrt{d_k}$ | $[B,H,S,S]$ | $[B,H,S,S]$ | in-place |
| Softmax | $[B,H,S,S]$ | $[B,H,S,S]$ | in-place |
| $PV$ matmul | $P: [B,H,S,S]$, $V: [B,H,S,d_h]$ | $[B,H,S,d_h]$ | 32 MB |

**Total intermediate memory:** 1+ GB (dominated by $QK^T$)

**This is why FlashAttention exists.** The $O(S^2)$ intermediate is the bottleneck.

## The Kernel Implication

**Naive kernel (what you should NOT do):**
```cuda
// Allocate O(S^2) intermediate in HBM
float* scores = allocate(B * H * S * S);  // 1 GB for LLaMA-3 8B!

// Kernel 1: QK^T matmul
attention_qk<<<...>>>(Q, K, scores);

// Kernel 2: Scale and softmax
attention_softmax<<<...>>>(scores, P);

// Kernel 3: PV matmul
attention_pv<<<...>>>(P, V, O);
```

**Problem:** Three separate kernel launches, each reading/writing the full $O(S^2)$ tensor from HBM. This is memory-bandwidth limited, not compute-limited.

**FlashAttention kernel (what you WILL implement):**
```cuda
// Single kernel, tile-by-tile computation
flash_attention<<<...>>>(Q, K, V, O) {
    // Load Q tile into SRAM
    // For each K, V tile:
    //   Load K tile into SRAM
    //   Load V tile into SRAM
    //   Compute QK^T tile (in SRAM, never written to HBM)
    //   Update running softmax max/sum (online softmax)
    //   Update output tile
    // Write O tile to HBM (once, at the end)
}
```

**Key insight:** The $O(S^2)$ intermediate never leaves SRAM. HBM traffic is $O(S)$ instead of $O(S^2)$.

## Numbers That Matter

| Model | B | H | S | d_h | $QK^T$ size (FP16) | $QK^T$ size (FP8) |
|-------|---|---|---|-----|-------------------|------------------|
| LLaMA-3 8B | 1 | 32 | 4096 | 128 | 1.0 GB | 0.5 GB |
| LLaMA-3 8B | 1 | 32 | 8192 | 128 | 4.0 GB | 2.0 GB |
| LLaMA-3 70B | 1 | 64 | 4096 | 128 | 2.0 GB | 1.0 GB |
| LLaMA-3 70B | 1 | 64 | 8192 | 128 | 8.0 GB | 4.0 GB |

**Note:** These are for a SINGLE sequence (B=1). At inference time, you cannot afford to materialize this.

## Common Interview Questions

**Q1: Derive why the $\sqrt{d_k}$ scaling factor is necessary. What happens if you omit it?**

<details>
<summary>Answer</summary>

Without scaling, the dot product $QK^T$ has variance proportional to $d_k$. Assuming $Q, K$ elements are i.i.d. with variance 1, the dot product of two $d_k$-dimensional vectors has variance $d_k$ (sum of $d_k$ independent products). Standard deviation is $\sqrt{d_k}$.

For $d_k = 128$, scores have std dev $\approx 11.3$. After softmax, $\exp(11.3) \approx 80,000$ and $\exp(-11.3) \approx 0$, causing the softmax to saturate (probabilities near 0 or 1). Saturated softmax has vanishing gradients: $\frac{\partial}{\partial x} \text{softmax}(x) \to 0$ as $|x| \to \infty$.

Dividing by $\sqrt{d_k}$ restores unit variance, keeping softmax in its linear regime where gradients flow.
</details>

**Q2: What are the tensor shapes at each step of attention for LLaMA-3 8B with batch=1, sequence=4096?**

<details>
<summary>Answer</summary>

- $Q, K, V$: $(1, 32, 4096, 128)$
- $QK^T$: $(1, 32, 4096, 4096)$ — 1 GB in FP16
- $QK^T / \sqrt{d_k}$: $(1, 32, 4096, 4096)$ — same
- $\text{softmax}(\cdot)$: $(1, 32, 4096, 4096)$ — same
- Output $PV$: $(1, 32, 4096, 128)$ — 32 MB
</details>

**Q3: Why is naive attention $O(S^2)$ in memory? Can you reduce this?**

<details>
<summary>Answer</summary>

The $QK^T$ operation produces a tensor of shape $[B, H, S, S]$. This is quadratic in sequence length because every query attends to every key (full self-attention).

You CAN reduce this with FlashAttention: compute $QK^T$ tile-by-tile in SRAM, never materializing the full $S \times S$ matrix in HBM. The output is still $O(S \cdot d_h)$, but intermediate HBM traffic is $O(S)$ instead of $O(S^2)$.

This is the core innovation of FlashAttention.
</details>

## Connection To Other Concepts

**Prerequisites:** None — this is the foundation. You need basic linear algebra (matrix multiplication, transpose) and probability (softmax).

**What this unlocks:** 
- Module 02 (KV Cache): Now that you know attention, you'll see why recomputing $K, V$ at every decode step is wasteful.
- Module 05 (FlashAttention): The $O(S^2)$ memory problem you just saw is exactly what FlashAttention solves.
- Module 10 (Arithmetic Intensity): You'll analyze whether attention is compute-bound or memory-bound.

**Next:** `02_causal_masking.md` — how to prevent attention from "cheating" by looking at future tokens.
