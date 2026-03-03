# Module 01: Attention — The Foundation

**One concept:** Scaled dot-product attention is the core operation of every transformer. Everything else (KV cache, FlashAttention, GQA) exists to make this operation feasible at inference time.

**Job mapping:** NVIDIA inference kernel engineer — you will implement this exact formula in your FlashAttention CuTe kernel. Every optimization you make must preserve the mathematical result.

---

## Files in This Module

Read in order:

1. **01_scaled_dot_product.md** — The core formula $softmax(QK^T / \sqrt{d_k})V$, why the $\sqrt{d_k}$ scaling exists, exact tensor shapes at every step.

2. **02_causal_masking.md** — Triangular mask for autoregressive generation, why prefill and decode have different mask requirements.

3. **03_multi_head_attention.md** — QKV projections, multi-head split/concat, complete FLOP count derivation.

4. **04_flop_and_memory_analysis.md** — Why attention is $O(S^2)$ in memory and compute, bandwidth bound proof for batch=1 decode.

5. **attention_forward.py** — Complete numpy implementation with shape assertions at every step. Run this before moving to Module 02.

---

## What You Must Be Able To Do After This Module

1. Write the attention formula from memory with correct shapes: $Q, K \in \mathbb{R}^{B \times H \times S \times d_h}$, output $\in \mathbb{R}^{B \times H \times S \times d_h}$

2. Explain why $\sqrt{d_k}$ scaling is necessary (gradient vanishing without it)

3. Derive the FLOP count: $4 B H S^2 d_h$ for single-head attention

4. Prove that at batch=1 decode, attention is memory-bandwidth bound, not compute-bound

5. Implement causal masking correctly (upper triangular with negative infinity)

---

## Before Moving To Module 02

Run `python attention_forward.py`. It must print `PASS`. If it prints `FAIL`, you do not understand the shapes yet.

**Next:** `01_scaled_dot_product.md` — the core formula
