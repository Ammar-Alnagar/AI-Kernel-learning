# Multi-Head Attention

## What This Is

Multi-head attention projects queries, keys, and values through learned linear transformations, splits them into multiple heads, applies attention independently per head, and concatenates the results. This allows the model to attend to different positions with different "representation subspaces."

**The complete formula:**
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

where
$$\text{head}_i = \text{Attention}(Q W_i^Q, K W_i^K, V W_i^V)$$

## Why A Kernel Engineer Needs This

**This is the complete forward pass you implement in your FlashAttention kernel.** The QKV projections are GEMM operations (often fused with the embedding lookup). The multi-head split is a reshape (no data movement). The output projection is another GEMM. Understanding the complete data flow — including where data is moved vs. where it is computed — is essential for kernel optimization.

**Interview relevance:** NVIDIA interviewers ask for the complete FLOP count of multi-head attention. You must be able to derive it from first principles, including all projections.

## The Math

### Step 1: Input and Projections

**Input:**
$$X \in \mathbb{R}^{B \times S \times d}$$

Where:
- $B$ = batch size
- $S$ = sequence length
- $d$ = model dimension (hidden size)

**Learned projections:**
$$W^Q, W^K, W^V \in \mathbb{R}^{d \times d}$$

**Projected Q, K, V:**
$$Q_{\text{raw}} = X W^Q \in \mathbb{R}^{B \times S \times d}$$
$$K_{\text{raw}} = X W^K \in \mathbb{R}^{B \times S \times d}$$
$$V_{\text{raw}} = X W^V \in \mathbb{R}^{B \times S \times d}$$

**Shape analysis:**
$$X: [B, S, d]$$
$$W^Q: [d, d]$$
$$Q_{\text{raw}} = X W^Q: [B, S, d]$$

**Worked example (LLaMA-3 8B):**
- $B = 1$, $S = 4096$, $d = 4096$
- $X$: $(1, 4096, 4096)$
- $W^Q$: $(4096, 4096)$
- $Q_{\text{raw}}$: $(1, 4096, 4096)$

### Step 2: Split Into Heads

Reshape from $[B, S, d]$ to $[B, H, S, d_h]$ where $d_h = d / H$.

$$Q = \text{reshape}(Q_{\text{raw}}, [B, S, d]) \to [B, H, S, d_h]$$
$$K = \text{reshape}(K_{\text{raw}}, [B, S, d]) \to [B, H, S, d_h]$$
$$V = \text{reshape}(V_{\text{raw}}, [B, S, d]) \to [B, H, S, d_h]$$

**This is a reshape, not a copy.** The data layout changes, but no data is moved (if the tensor is contiguous and the reshape is compatible).

**Worked example (LLaMA-3 8B):**
- $d = 4096$, $H = 32$, $d_h = 4096 / 32 = 128$
- $Q_{\text{raw}}$: $(1, 4096, 4096)$
- $Q$: $(1, 32, 4096, 128)$

**Memory layout note:** In PyTorch, `view(B, S, H, d_h).transpose(1, 2)` achieves this. In CuTe, you define a layout that interprets the flat memory as $[B, H, S, d_h]$.

### Step 3: Per-Head Attention

Apply scaled dot-product attention independently for each head:

$$\text{head}_i = \text{Attention}(Q_i, K_i, V_i) = \text{softmax}\left(\frac{Q_i K_i^T}{\sqrt{d_h}}\right) V_i$$

Where $Q_i, K_i, V_i$ are the slices for head $i$.

**Shape per head:**
$$Q_i: [B, S, d_h]$$
$$K_i: [B, S, d_h]$$
$$V_i: [B, S, d_h]$$
$$\text{head}_i: [B, S, d_h]$$

**Worked example:**
- $Q_i$: $(1, 4096, 128)$
- $K_i$: $(1, 4096, 128)$
- $\text{head}_i$: $(1, 4096, 128)$

### Step 4: Concatenate Heads

$$\text{Concat}(\text{head}_1, \ldots, \text{head}_H) \in \mathbb{R}^{B \times S \times d}$$

This is the inverse of Step 2: reshape from $[B, H, S, d_h]$ back to $[B, S, d]$.

**Shape:**
$$\text{Concat}(\cdot): [B, H, S, d_h] \to [B, S, d]$$

**Worked example:**
- Input: $(1, 32, 4096, 128)$
- Output: $(1, 4096, 4096)$

### Step 5: Output Projection

$$O = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W^O$$

Where $W^O \in \mathbb{R}^{d \times d}$ is a learned projection matrix.

**Shape:**
$$\text{Concat}: [B, S, d]$$
$$W^O: [d, d]$$
$$O: [B, S, d]$$

**Worked example:**
- Concat: $(1, 4096, 4096)$
- $W^O$: $(4096, 4096)$
- $O$: $(1, 4096, 4096)$

### Step 6: Complete Multi-Head Attention Flow

$$X \xrightarrow{W^Q, W^K, W^V} Q_{\text{raw}}, K_{\text{raw}}, V_{\text{raw}} \xrightarrow{\text{reshape}} Q, K, V \xrightarrow{\text{Attention}} \text{heads} \xrightarrow{\text{concat}} \text{Concat} \xrightarrow{W^O} O$$

**Complete shape flow (LLaMA-3 8B):**
$$X: (1, 4096, 4096)$$
$$Q_{\text{raw}}, K_{\text{raw}}, V_{\text{raw}}: (1, 4096, 4096)$$
$$Q, K, V: (1, 32, 4096, 128)$$
$$\text{heads}: (1, 32, 4096, 128)$$
$$\text{Concat}: (1, 4096, 4096)$$
$$O: (1, 4096, 4096)$$

## FLOP Count Derivation

### Projection FLOPs

**Q projection:** $Q_{\text{raw}} = X W^Q$
- Matrix multiply: $[B, S, d] \times [d, d] \to [B, S, d]$
- FLOPs: $2 \cdot B \cdot S \cdot d \cdot d = 2 B S d^2$

**K projection:** Same, $2 B S d^2$ FLOPs

**V projection:** Same, $2 B S d^2$ FLOPs

**Total projection FLOPs:** $6 B S d^2$

**Worked example (LLaMA-3 8B, B=1, S=4096, d=4096):**
$$6 \cdot 1 \cdot 4096 \cdot 4096^2 = 6 \cdot 4096 \cdot 16,777,216 = 412,316,860,416 \approx 412 \text{ GFLOPs}$$

### Attention FLOPs (per head)

From Module 01.4, single-head attention FLOPs:
- $QK^T$ matmul: $2 \cdot S \cdot S \cdot d_h$ FLOPs
- $PV$ matmul: $2 \cdot S \cdot S \cdot d_h$ FLOPs
- Total per head: $4 S^2 d_h$ FLOPs

### Attention FLOPs (all heads)

$$H \cdot 4 S^2 d_h = 4 S^2 (H \cdot d_h) = 4 S^2 d$$

Since $H \cdot d_h = d$.

**Worked example (LLaMA-3 8B, S=4096, d=4096):**
$$4 \cdot 4096^2 \cdot 4096 = 4 \cdot 16,777,216 \cdot 4096 = 274,877,906,944 \approx 275 \text{ GFLOPs}$$

### Output Projection FLOPs

$O = \text{Concat} \cdot W^O$
- Matrix multiply: $[B, S, d] \times [d, d] \to [B, S, d]$
- FLOPs: $2 B S d^2$

**Worked example:**
$$2 \cdot 1 \cdot 4096 \cdot 4096^2 = 137,438,953,472 \approx 137 \text{ GFLOPs}$$

### Total Multi-Head Attention FLOPs

$$\text{Total} = 6 B S d^2 + 4 B S^2 d + 2 B S d^2 = 8 B S d^2 + 4 B S^2 d$$

**Factored:**
$$\text{Total} = 4 B S d (2d + S)$$

**Worked example (LLaMA-3 8B, B=1, S=4096, d=4096):**
$$4 \cdot 1 \cdot 4096 \cdot 4096 \cdot (2 \cdot 4096 + 4096) = 4 \cdot 4096^2 \cdot 3 \cdot 4096 = 206,158,430,208 \approx 206 \text{ GFLOPs}$$

Wait, let me recalculate:
$$8 B S d^2 + 4 B S^2 d$$
$$= 8 \cdot 1 \cdot 4096 \cdot 4096^2 + 4 \cdot 1 \cdot 4096^2 \cdot 4096$$
$$= 8 \cdot 4096^3 + 4 \cdot 4096^3 = 12 \cdot 4096^3$$
$$= 12 \cdot 68,719,476,736 = 824,633,720,832 \approx 825 \text{ GFLOPs}$$

**Breakdown:**
- Projections (Q, K, V): 412 GFLOPs (50%)
- Attention ($QK^T$, $PV$): 275 GFLOPs (33%)
- Output projection: 137 GFLOPs (17%)
- **Total:** 825 GFLOPs

**Key insight:** At $S = d = 4096$, projections dominate (67% of FLOPs). At $S \gg d$, attention dominates.

## Shapes and Sizes

| Operation | Input shapes | Output shape | FLOPs (LLaMA-3 8B, B=1, S=4096) |
|-----------|--------------|--------------|----------------------------------|
| Q projection | $X: [B,S,d]$, $W^Q: [d,d]$ | $[B,S,d]$ | 137 GFLOPs |
| K projection | $X: [B,S,d]$, $W^K: [d,d]$ | $[B,S,d]$ | 137 GFLOPs |
| V projection | $X: [B,S,d]$, $W^V: [d,d]$ | $[B,S,d]$ | 137 GFLOPs |
| Reshape to heads | $[B,S,d]$ | $[B,H,S,d_h]$ | 0 FLOPs (reshape) |
| $QK^T$ (all heads) | $Q: [B,H,S,d_h]$, $K: [B,H,S,d_h]$ | $[B,H,S,S]$ | 137 GFLOPs |
| $PV$ (all heads) | $P: [B,H,S,S]$, $V: [B,H,S,d_h]$ | $[B,H,S,d_h]$ | 137 GFLOPs |
| Concat heads | $[B,H,S,d_h]$ | $[B,S,d]$ | 0 FLOPs (reshape) |
| Output projection | $[B,S,d]$, $W^O: [d,d]$ | $[B,S,d]$ | 137 GFLOPs |
| **Total** | | | **825 GFLOPs** |

## The Kernel Implication

### Fusion Opportunities

**QKV projection fusion:** Instead of three separate GEMMs for $W^Q, W^K, W^V$, stack them:

$$W^{QKV} = \begin{bmatrix} W^Q \\ W^K \\ W^V \end{bmatrix} \in \mathbb{R}^{3d \times d}$$

$$QKV_{\text{raw}} = X W^{QKV} \in \mathbb{R}^{B \times S \times 3d}$$

Then split into $Q, K, V$ via reshape/slice.

**Kernel benefit:** One GEMM instead of three. Better occupancy, less kernel launch overhead.

**In CuTe:**
```cpp
// Fused QKV projection
auto qkv = gemm(X, W_QKV_stacked);  // [B, S, 3d]

// Split into Q, K, V (no copy, just views)
auto Q = qkv.slice(0, d);    // [B, S, d]
auto K = qkv.slice(d, 2*d);  // [B, S, d]
auto V = qkv.slice(2*d, 3*d);// [B, S, d]

// Reshape to multi-head
auto Q_heads = Q.view(B, S, H, d_h).transpose(1, 2);  // [B, H, S, d_h]
```

### Memory Bandwidth Analysis

**Data movement (LLaMA-3 8B, B=1, S=4096, FP16):**

**Reads:**
- $X$: $B \cdot S \cdot d \cdot 2 = 1 \cdot 4096 \cdot 4096 \cdot 2 = 32$ MB
- $W^Q, W^K, W^V$: $3 \cdot d \cdot d \cdot 2 = 3 \cdot 4096^2 \cdot 2 = 96$ MB
- $W^O$: $d \cdot d \cdot 2 = 32$ MB
- **Total reads:** 160 MB

**Writes:**
- $O$: $B \cdot S \cdot d \cdot 2 = 32$ MB
- **Total writes:** 32 MB

**Total HBM traffic:** 192 MB

**Arithmetic intensity:**
$$\text{AI} = \frac{825 \times 10^9 \text{ FLOPs}}{192 \times 10^6 \text{ bytes}} \approx 4297 \text{ FLOPs/byte}$$

**This is compute-bound on H100** (peak ~2000 FLOPs/byte for FP16). However, this analysis is for the entire attention layer. Individual kernels (like $QK^T$ matmul) may be memory-bound.

## Numbers That Matter

| Model | B | S | d | H | d_h | Total FLOPs |
|-------|---|---|---|---|-----|-------------|
| LLaMA-3 8B | 1 | 4096 | 4096 | 32 | 128 | 825 GFLOPs |
| LLaMA-3 8B | 1 | 8192 | 4096 | 32 | 128 | 1.5 TFLOPs |
| LLaMA-3 70B | 1 | 4096 | 8192 | 64 | 128 | 3.3 TFLOPs |
| LLaMA-3 70B | 1 | 8192 | 8192 | 64 | 128 | 6.0 TFLOPs |

**Note:** FLOPs scale linearly with $B$, quadratically with $S$, and quadratically with $d$.

## Common Interview Questions

**Q1: Derive the complete FLOP count for multi-head attention including all projections.**

<details>
<summary>Answer</summary>

Q, K, V projections: $3 \times 2 B S d^2 = 6 B S d^2$

Attention (all heads): $H \times 4 S^2 d_h = 4 S^2 d$ (since $H \cdot d_h = d$)

Output projection: $2 B S d^2$

Total: $6 B S d^2 + 4 B S^2 d + 2 B S d^2 = 8 B S d^2 + 4 B S^2 d = 4 B S d (2d + S)$

For LLaMA-3 8B (B=1, S=4096, d=4096): ~825 GFLOPs
</details>

**Q2: What is the shape transformation from input to output in multi-head attention?**

<details>
<summary>Answer</summary>

$X: [B, S, d]$ (input)

$Q_{\text{raw}}, K_{\text{raw}}, V_{\text{raw}}: [B, S, d]$ (after projections)

$Q, K, V: [B, H, S, d_h]$ (after reshape/split)

$\text{heads}: [B, H, S, d_h]$ (after attention)

$\text{Concat}: [B, S, d]$ (after concatenate)

$O: [B, S, d]$ (after output projection)
</details>

**Q3: When is multi-head attention compute-bound vs. memory-bound?**

<details>
<summary>Answer</summary>

Compute-bound when arithmetic intensity > hardware roofline (~2000 FLOPs/byte for H100 FP16).

AI = FLOPs / bytes = $(8 B S d^2 + 4 B S^2 d) / (4 B S d + 4 d^2)$ (approximate, ignoring weight reuse)

For large $S$ and $d$, AI is high (compute-bound). For small batch sizes and short sequences, AI is low (memory-bound).

At decode time (S=1 for new token, but attending to S=4096 cached), the $QK^T$ operation is memory-bound because it reads 4096 keys but only computes 4096 scores.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01.1 (scaled dot-product attention) — you need the base formula.

**What this unlocks:**
- Module 03 (Attention Variants): GQA and MQA modify the multi-head structure.
- Module 05 (FlashAttention): The tile loop operates on the $[B, H, S, d_h]$ tensors you just analyzed.
- Module 10 (Arithmetic Intensity): You now have the FLOP count needed for roofline analysis.

**Next:** `04_flop_and_memory_analysis.md` — detailed $O(S^2)$ analysis and bandwidth bound proof.
