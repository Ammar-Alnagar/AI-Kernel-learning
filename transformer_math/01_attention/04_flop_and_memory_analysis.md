# FLOP and Memory Analysis

## What This Is

This is the complete computational and memory complexity analysis of attention. We derive why attention is $O(S^2)$ in both FLOPs and memory, prove that decode at batch=1 is memory-bandwidth bound, and establish the arithmetic intensity formulas used throughout this directory.

**The key results:**
- Attention FLOPs: $4 B H S^2 d_h$ (for the $QK^T$ and $PV$ matmuls)
- Attention memory (intermediate): $B H S^2$ elements ($O(S^2)$)
- Decode at batch=1 is **always** memory-bandwidth bound
- Prefill can be compute-bound at large batch sizes

## Why A Kernel Engineer Needs This

**This analysis determines whether your kernel should optimize for compute or memory bandwidth.** If an operation is memory-bound, you optimize for coalesced loads, cache reuse, and minimizing HBM traffic. If compute-bound, you optimize for occupancy, instruction-level parallelism, and tensor core utilization.

**Interview relevance:** Modular and Cerebras interviewers ask candidates to analyze whether a given operation is compute-bound or memory-bound. You must be able to compute arithmetic intensity and compare against hardware rooflines.

## The Math

### FLOP Count: Single-Head Attention

**Operation:** $\text{Attention}(Q, K, V) = \text{softmax}(QK^T / \sqrt{d_k}) V$

**Shapes:**
$$Q: [B, S, d_h], \quad K: [B, S, d_h], \quad V: [B, S, d_h]$$

### $QK^T$ Matmul FLOPs

$$QK^T: [B, S, d_h] \times [B, d_h, S] \to [B, S, S]$$

Matrix multiplication FLOPs: $2 \cdot \text{rows} \cdot \text{cols} \cdot \text{inner}$

$$\text{FLOPs}_{QK^T} = 2 \cdot B \cdot S \cdot S \cdot d_h = 2 B S^2 d_h$$

**Worked example (B=1, S=4096, d_h=128):**
$$2 \cdot 1 \cdot 4096^2 \cdot 128 = 2 \cdot 16,777,216 \cdot 128 = 4,294,967,296 \approx 4.3 \text{ GFLOPs}$$

### Softmax FLOPs

Softmax over $S$ elements for each of $B \cdot S$ rows:
- Exponential: $B \cdot S \cdot S$ elements
- Sum: $B \cdot S$ reductions, each summing $S$ elements $\approx B \cdot S^2$ operations
- Division: $B \cdot S \cdot S$ elements

$$\text{FLOPs}_{\text{softmax}} \approx 3 B S^2$$

**Worked example:**
$$3 \cdot 1 \cdot 4096^2 = 3 \cdot 16,777,216 = 50,331,648 \approx 50 \text{ MFLOPs}$$

**Note:** Softmax FLOPs are negligible compared to matmuls (< 1%).

### $PV$ Matmul FLOPs

$$PV: [B, S, S] \times [B, S, d_h] \to [B, S, d_h]$$

$$\text{FLOPs}_{PV} = 2 \cdot B \cdot S \cdot d_h \cdot S = 2 B S^2 d_h$$

**Worked example:**
$$2 \cdot 1 \cdot 4096^2 \cdot 128 = 4.3 \text{ GFLOPs (same as } QK^T)$$

### Total Single-Head Attention FLOPs

$$\text{FLOPs}_{\text{single}} = 2 B S^2 d_h + 3 B S^2 + 2 B S^2 d_h = 4 B S^2 d_h + 3 B S^2$$

For $d_h \gg 3$ (always true, $d_h \geq 64$), the $3 B S^2$ term is negligible:

$$\boxed{\text{FLOPs}_{\text{single}} \approx 4 B S^2 d_h}$$

### Multi-Head Attention FLOPs

Multiply by $H$ heads:

$$\text{FLOPs}_{\text{multi}} = H \cdot 4 B S^2 d_h = 4 B S^2 (H d_h) = 4 B S^2 d$$

Since $H d_h = d$ (total model dimension).

**Worked example (LLaMA-3 8B, B=1, S=4096, d=4096):**
$$4 \cdot 1 \cdot 4096^2 \cdot 4096 = 4 \cdot 16,777,216 \cdot 4096 = 274,877,906,944 \approx 275 \text{ GFLOPs}$$

### Memory Complexity: The $O(S^2)$ Wall

**Intermediate tensors:**

| Tensor | Shape | Elements | Bytes (FP16) |
|--------|-------|----------|--------------|
| $QK^T$ scores | $[B, H, S, S]$ | $B H S^2$ | $2 B H S^2$ |
| Scaled scores | $[B, H, S, S]$ | $B H S^2$ | $2 B H S^2$ (in-place) |
| Attention probs $P$ | $[B, H, S, S]$ | $B H S^2$ | $2 B H S^2$ |

**Total intermediate memory:** $O(B H S^2)$ elements, $O(B H S^2 \cdot \text{dtype\_bytes})$ bytes.

**Worked example (LLaMA-3 8B, B=1, S=4096, H=32, FP16):**
$$\text{Elements} = 1 \cdot 32 \cdot 4096^2 = 32 \cdot 16,777,216 = 536,870,912$$
$$\text{Bytes} = 536,870,912 \cdot 2 = 1,073,741,824 \approx 1 \text{ GB}$$

**This is the memory wall.** A single sequence at $S=4096$ requires 1 GB of intermediate storage. At $S=8192$, it's 4 GB. This is why naive attention is infeasible at production scale.

### Memory Bandwidth Analysis

**HBM traffic for naive attention:**

**Reads:**
- $Q$: $B H S d_h \cdot 2$ bytes
- $K$: $B H S d_h \cdot 2$ bytes
- $V$: $B H S d_h \cdot 2$ bytes
- **Total:** $6 B H S d_h \cdot 2 = 12 B H S d_h$ bytes

**Writes:**
- $QK^T$ scores: $B H S^2 \cdot 2$ bytes
- Attention probs $P$: $B H S^2 \cdot 2$ bytes (or in-place)
- Output $O$: $B H S d_h \cdot 2$ bytes
- **Total:** $(2 B H S^2 + B H S d_h) \cdot 2$ bytes (assuming in-place softmax)

**Total HBM traffic:**
$$\text{Bytes}_{\text{naive}} = 12 B H S d_h + 4 B H S^2 + 2 B H S d_h = 14 B H S d_h + 4 B H S^2$$

**Factored:**
$$\text{Bytes}_{\text{naive}} = 2 B H S (7 d_h + 2 S)$$

**Worked example (LLaMA-3 8B, B=1, S=4096, H=32, d_h=128):**
$$2 \cdot 1 \cdot 32 \cdot 4096 \cdot (7 \cdot 128 + 2 \cdot 4096)$$
$$= 262,144 \cdot (896 + 8192) = 262,144 \cdot 9088 = 2,382,359,552 \approx 2.4 \text{ GB}$$

### Arithmetic Intensity

**Definition:** Arithmetic intensity (AI) = FLOPs / bytes

$$\text{AI}_{\text{naive}} = \frac{4 B H S^2 d_h}{2 B H S (7 d_h + 2 S)} = \frac{4 S d_h}{2 (7 d_h + 2 S)} = \frac{2 S d_h}{7 d_h + 2 S}$$

**Simplified:**
$$\boxed{\text{AI}_{\text{naive}} = \frac{2 S d_h}{7 d_h + 2 S}}$$

**Worked example (LLaMA-3 8B, S=4096, d_h=128):**
$$\text{AI} = \frac{2 \cdot 4096 \cdot 128}{7 \cdot 128 + 2 \cdot 4096} = \frac{1,048,576}{896 + 8192} = \frac{1,048,576}{9088} \approx 115 \text{ FLOPs/byte}$$

**Interpretation:** 115 FLOPs/byte is **memory-bound** on H100 (peak ~2000 FLOPs/byte for FP16 tensor cores).

### Decode vs. Prefill: Different Characteristics

**Prefill:** Process all $S$ tokens in parallel.
- $Q, K, V$ all have shape $[B, H, S, d_h]$
- Full $QK^T$ matmul: $O(S^2)$
- Can be compute-bound at large $B$

**Decode:** Generate one token at a time.
- $Q$: $[B, H, 1, d_h]$ (single query)
- $K, V$: $[B, H, S, d_h]$ (cached, all previous tokens)
- $QK^T$: $[B, H, 1, S]$ (linear in $S$, not quadratic!)

**Decode FLOPs:**
$$\text{FLOPs}_{\text{decode}} = 2 \cdot B \cdot H \cdot 1 \cdot S \cdot d_h + 2 \cdot B \cdot H \cdot 1 \cdot S \cdot d_h = 4 B H S d_h$$

**Decode HBM traffic (with KV cache):**
- Read $Q$: $B H d_h \cdot 2$ bytes
- Read $K_{\text{cache}}$: $B H S d_h \cdot 2$ bytes
- Read $V_{\text{cache}}$: $B H S d_h \cdot 2$ bytes
- Write $O$: $B H d_h \cdot 2$ bytes
- **Total:** $2 B H d_h + 4 B H S d_h = 2 B H d_h (1 + 2S)$ bytes

**Decode arithmetic intensity:**
$$\text{AI}_{\text{decode}} = \frac{4 B H S d_h}{2 B H d_h (1 + 2S)} = \frac{4 S}{2 (1 + 2S)} = \frac{2 S}{1 + 2S}$$

For $S \gg 1$:
$$\text{AI}_{\text{decode}} \approx \frac{2 S}{2 S} = 1 \text{ FLOP/byte}$$

**This is extremely memory-bound.** At $S = 4096$:
$$\text{AI}_{\text{decode}} = \frac{2 \cdot 4096}{1 + 2 \cdot 4096} = \frac{8192}{8193} \approx 1.0 \text{ FLOP/byte}$$

**Compare to H100 roofline:** ~2000 FLOPs/byte (FP16 tensor cores)

**Conclusion:** Decode is **2000x more memory-bound** than the hardware can handle. The GPU is severely underutilized.

## Shapes and Sizes

| Operation | Prefill shapes | Decode shapes |
|-----------|----------------|---------------|
| $Q$ | $[B, H, S, d_h]$ | $[B, H, 1, d_h]$ |
| $K, V$ | $[B, H, S, d_h]$ | $[B, H, S, d_h]$ (cached) |
| $QK^T$ | $[B, H, S, S]$ | $[B, H, 1, S]$ |
| $P$ | $[B, H, S, S]$ | $[B, H, 1, S]$ |
| Output | $[B, H, S, d_h]$ | $[B, H, 1, d_h]$ |
| FLOPs | $4 B H S^2 d_h$ | $4 B H S d_h$ |
| AI (H100) | ~115 FLOPs/byte | ~1 FLOP/byte |

## The Kernel Implication

### Why Decode Is Always Bandwidth-Bound

At decode time:
1. You load $S$ cached keys and values from HBM
2. You compute $S$ dot products (one per cached key)
3. You produce 1 output token

**The compute-to-memory ratio is $O(1)$:** constant FLOPs per byte loaded.

**This is fundamental to autoregressive generation.** No kernel optimization can change this — the algorithm itself is memory-bound. The only solutions are:
1. Increase batch size (process multiple sequences in parallel)
2. Use KV cache quantization (reduce bytes per element)
3. Use larger on-chip caches (HBM is the bottleneck)

### FlashAttention: Reducing HBM Traffic

FlashAttention reduces HBM traffic from $O(B H S^2)$ to $O(B H S d_h)$ by:
- Never writing $QK^T$ scores to HBM (keep in SRAM)
- Never writing attention probs $P$ to HBM (keep in SRAM)
- Writing output $O$ to HBM only once (not per-tile)

**FlashAttention HBM traffic:**
$$\text{Bytes}_{\text{FA}} = O(B H S d_h)$$

**FlashAttention arithmetic intensity:**
$$\text{AI}_{\text{FA}} = \frac{4 B H S^2 d_h}{O(B H S d_h)} = O(S)$$

For $S = 4096$, AI increases from ~115 to ~4000+ FLOPs/byte, becoming **compute-bound**.

**This is why FlashAttention matters:** it shifts attention from memory-bound to compute-bound, enabling full GPU utilization.

## Numbers That Matter

| Scenario | Model | B | S | AI (FLOPs/byte) | Bound |
|----------|-------|---|---|-----------------|-------|
| Prefill | LLaMA-3 8B | 1 | 4096 | 115 | Memory |
| Prefill | LLaMA-3 8B | 32 | 4096 | 3680 | Compute |
| Prefill | LLaMA-3 8B | 1 | 8192 | 230 | Memory |
| Decode | LLaMA-3 8B | 1 | 4096 | 1.0 | Memory |
| Decode | LLaMA-3 8B | 128 | 4096 | 128 | Memory |
| Decode | LLaMA-3 8B | 512 | 4096 | 512 | Memory |

**Note:** Decode remains memory-bound even at large batch sizes. Prefill becomes compute-bound at moderate batch sizes.

## Common Interview Questions

**Q1: Prove that decode at batch=1 is memory-bandwidth bound.**

<details>
<summary>Answer</summary>

Decode FLOPs: $4 B H S d_h$ (for $QK^T$ and $PV$ with $Q$ having sequence length 1)

Decode HBM traffic (with KV cache): $2 B H d_h + 4 B H S d_h \approx 4 B H S d_h$ bytes (for $S \gg 1$)

Arithmetic intensity: $\text{AI} = \frac{4 B H S d_h}{4 B H S d_h} = 1$ FLOP/byte

H100 peak AI (FP16 tensor cores): ~2000 FLOPs/byte

Since $1 \ll 2000$, decode is memory-bandwidth bound. The GPU spends most cycles waiting for memory, not computing.
</details>

**Q2: Why is attention $O(S^2)$ in memory? What is the specific tensor?**

<details>
<summary>Answer</summary>

The $QK^T$ operation produces a tensor of shape $[B, H, S, S]$. This has $B H S^2$ elements, quadratic in sequence length.

For LLaMA-3 8B (B=1, H=32, S=4096), this is 536 million elements, or 1 GB in FP16.

This intermediate tensor must be stored in HBM in naive implementations, making attention infeasible at long sequences.

FlashAttention avoids this by computing $QK^T$ tile-by-tile in SRAM, never materializing the full $S \times S$ matrix in HBM.
</details>

**Q3: At what batch size does LLaMA-3 8B prefill become compute-bound on H100?**

<details>
<summary>Answer</summary>

From the AI formula: $\text{AI} = \frac{2 S d_h}{7 d_h + 2 S}$ for single-head, scaled by $B$ for multi-head.

For multi-head with batch $B$: $\text{AI} = B \cdot \frac{2 S d_h}{7 d_h + 2 S}$

Set AI = 2000 (H100 roofline) and solve for $B$:
$$B \cdot \frac{2 \cdot 4096 \cdot 128}{7 \cdot 128 + 2 \cdot 4096} = 2000$$
$$B \cdot 115 = 2000$$
$$B \approx 17.4$$

At batch size ≥ 18, LLaMA-3 8B prefill at S=4096 becomes compute-bound on H100.
</details>

## Connection To Other Concepts

**Prerequisites:** Module 01.3 (multi-head attention) — you need the FLOP count foundation.

**What this unlocks:**
- Module 02 (KV Cache): Now you understand why decode is memory-bound — KV cache is essential.
- Module 05 (FlashAttention): The $O(S^2)$ memory problem is what FlashAttention solves.
- Module 10 (Arithmetic Intensity): This is the foundation for roofline analysis.

**Next:** Run `attention_forward.py` to see these formulas in code. Then move to Module 02 (KV Cache).
