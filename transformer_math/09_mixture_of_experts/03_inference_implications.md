# Inference Implications

## What This Is

MoE inference requires all-to-all communication for expert parallelism. This adds communication overhead but enables larger models.

## The Math

### Communication Volume

**For batch size $B$, sequence length $S$, experts $E$, GPUs $G$:**

**Tokens per GPU:** $\frac{B \cdot S}{G}$

**Tokens per expert (uniform):** $\frac{B \cdot S \cdot k}{E}$

**All-to-all volume (per GPU):**
$$\text{Volume} = \frac{B \cdot S \cdot d}{G} \cdot \text{dtype\_bytes}$$

(Send all tokens, receive expert outputs)

### Compute vs. Communication

**Compute time (per expert FFN):**
$$T_{\text{compute}} = \frac{2 \cdot (B \cdot S \cdot k / E) \cdot d \cdot d_{ff}}{\text{Peak FLOPs}}$$

**Communication time:**
$$T_{\text{comm}} = \frac{B \cdot S \cdot d \cdot \text{dtype\_bytes}}{\text{BW}}$$

**Overlap:** Modern systems overlap compute and communication.

## Numbers That Matter

| Model | E | k | GPUs | Comm Overhead |
|-------|---|---|------|---------------|
| Mixtral 8x7B | 8 | 2 | 8 | ~10% |
| Grok-1 | 64 | 2 | 64 | ~20% |

## The Kernel Implication

**NCCL all-to-all:**
```cpp
// Dispatch tokens to experts
ncclGroupStart();
for (int gpu = 0; gpu < num_gpus; ++gpu) {
    ncclSend(tokens_for_gpu[gpu], ...);
    ncclRecv(tokens_from_gpu[gpu], ...);
}
ncclGroupEnd();
```
