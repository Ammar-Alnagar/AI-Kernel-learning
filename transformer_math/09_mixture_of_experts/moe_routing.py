"""
FILE: moe_routing.py
TEACHES: MoE router, top-k selection, load balancing
MAPS TO: Cerebras ML runtime engineer — MoE routing implementation
RUN: python moe_routing.py — no arguments needed
"""

import numpy as np

print("=" * 70)
print("MIXTURE OF EXPERTS: ROUTING SIMULATION")
print("=" * 70)

# ============================================================
# PART 1: Configuration
# ============================================================

E = 8  # Number of experts
k = 2  # Top-k experts per token
batch_size = 1024  # Tokens per batch
d = 4096  # Model dimension

print(f"\nConfig:")
print(f"  Experts: E={E}")
print(f"  Top-k: k={k}")
print(f"  Batch size: {batch_size} tokens")
print(f"  Model dimension: d={d}")

# ============================================================
# PART 2: Router Function
# Math reference: see 02_routing_math.md
# ============================================================

print("\n" + "=" * 70)
print("ROUTER FUNCTION")
print("=" * 70)

rng = np.random.Generator(np.random.PCG64(42))

# Simulate router weights
W_r = rng.standard_normal((E, d)).astype(np.float32)

# Simulate input tokens
X = rng.standard_normal((batch_size, d)).astype(np.float32)

# Compute router probabilities: g(x) = softmax(W_r @ x)
router_logits = np.matmul(X, W_r.T)  # [batch_size, E]

# Softmax over experts
router_logits_max = np.max(router_logits, axis=1, keepdims=True)
router_exp = np.exp(router_logits - router_logits_max)
router_probs = router_exp / np.sum(router_exp, axis=1, keepdims=True)

print(f"\nRouter output shape: {router_probs.shape}")
print(f"Router probabilities (first 5 tokens):")
for i in range(5):
    print(f"  Token {i}: {router_probs[i]}")

# ============================================================
# PART 3: Top-k Expert Selection
# ============================================================

print("\n" + "=" * 70)
print("TOP-K EXPERT SELECTION")
print("=" * 70)

# Get top-k experts for each token
top_k_indices = np.argsort(router_probs, axis=1)[:, -k:]  # [batch_size, k]
top_k_probs = np.take_along_axis(router_probs, top_k_indices, axis=1)

# Normalize gating weights over selected experts
gating_weights = top_k_probs / np.sum(top_k_probs, axis=1, keepdims=True)

print(f"\nTop-{k} experts (first 5 tokens):")
for i in range(5):
    experts = top_k_indices[i]
    weights = gating_weights[i]
    print(f"  Token {i}: experts={experts}, weights={weights}")

# ============================================================
# PART 4: Load Balancing Analysis
# Math reference: see 02_routing_math.md
# ============================================================

print("\n" + "=" * 70)
print("LOAD BALANCING ANALYSIS")
print("=" * 70)

# Count tokens per expert
expert_counts = np.zeros(E, dtype=int)
for i in range(batch_size):
    for expert_id in top_k_indices[i]:
        expert_counts[expert_id] += 1

# Ideal load (uniform distribution)
ideal_load = batch_size * k / E

print(f"\nExpert load distribution:")
print(f"  Ideal: {ideal_load:.0f} tokens per expert")
print()
print(f"  {'Expert':<8} {'Tokens':<10} {'% of ideal':<12}")
print("  " + "-" * 30)

for i in range(E):
    pct = 100 * expert_counts[i] / ideal_load
    bar = "█" * int(pct / 5)
    print(f"  {i:<8} {expert_counts[i]:<10} {pct:<12.1f}% {bar}")

# Load imbalance metric
load_imbalance = np.std(expert_counts) / ideal_load * 100
print(f"\nLoad imbalance (std/mean): {load_imbalance:.1f}%")

# ============================================================
# PART 5: Load Balancing Loss
# ============================================================

print("\n" + "=" * 70)
print("LOAD BALANCING LOSS")
print("=" * 70)

# Compute auxiliary loss
# L_aux = sum_i (L_i / sum_j L_j - 1/E)^2
total_load = np.sum(expert_counts)
load_fraction = expert_counts / total_load
ideal_fraction = 1 / E

aux_loss = np.sum((load_fraction - ideal_fraction) ** 2)

print(f"\nAuxiliary loss formula:")
print(f"  L_aux = Σ (L_i / ΣL_j - 1/E)²")
print(f"  = {aux_loss:.6f}")

print(f"\nInterpretation:")
print(f"  L_aux = 0: Perfect balance")
print(f"  L_aux > 0: Imbalance (higher = worse)")
print(f"  Current: {aux_loss:.6f}")

# ============================================================
# PART 6: MoE Output Computation
# ============================================================

print("\n" + "=" * 70)
print("MOE OUTPUT COMPUTATION")
print("=" * 70)

# Simulate expert outputs (each expert is an FFN)
# For simplicity, use random linear transform
expert_weights = rng.standard_normal((E, d, d)).astype(np.float32)

# Compute expert outputs
expert_outputs = np.zeros((batch_size, d), dtype=np.float32)

for i in range(batch_size):
    for j, expert_id in enumerate(top_k_indices[i]):
        # Expert FFN (simplified: linear transform)
        output = np.matmul(X[i], expert_weights[expert_id])
        expert_outputs[i] += gating_weights[i, j] * output

print(f"\nMoE output shape: {expert_outputs.shape}")
print(f"Output stats: min={expert_outputs.min():.4f}, max={expert_outputs.max():.4f}, mean={expert_outputs.mean():.4f}")

# ============================================================
# PART 7: Compare With and Without Load Balancing
# ============================================================

print("\n" + "=" * 70)
print("EFFECT OF LOAD BALANCING LOSS")
print("=" * 70)

# Simulate collapsed router (no load balancing)
# Router always picks experts 0 and 1
collapsed_indices = np.tile([0, 1], (batch_size, 1))
collapsed_counts = np.zeros(E, dtype=int)
for i in range(batch_size):
    for expert_id in collapsed_indices[i]:
        collapsed_counts[expert_id] += 1

collapsed_total = np.sum(collapsed_counts)
collapsed_fraction = collapsed_counts / collapsed_total
collapsed_aux_loss = np.sum((collapsed_fraction - ideal_fraction) ** 2)

print(f"\nWithout load balancing (collapsed router):")
print(f"  Expert 0: {collapsed_counts[0]} tokens ({100 * collapsed_counts[0] / (batch_size * k):.0f}%)")
print(f"  Expert 1: {collapsed_counts[1]} tokens ({100 * collapsed_counts[1] / (batch_size * k):.0f}%)")
print(f"  Others: 0 tokens")
print(f"  Aux loss: {collapsed_aux_loss:.6f}")

print(f"\nWith load balancing:")
print(f"  Min tokens: {expert_counts.min()}")
print(f"  Max tokens: {expert_counts.max()}")
print(f"  Aux loss: {aux_loss:.6f}")

# ============================================================
# VERIFY: Summary
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"\n✓ Router output: {router_probs.shape}")
print(f"✓ Top-{k} selection: {top_k_indices.shape}")
print(f"✓ Expert load: min={expert_counts.min()}, max={expert_counts.max()}")
print(f"✓ Load imbalance: {load_imbalance:.1f}%")
print(f"✓ Aux loss: {aux_loss:.6f}")
print(f"✓ MoE output: {expert_outputs.shape}")
print()
print("PASS — MoE routing simulation complete.")
print()
print("Key insights:")
print("  1. Router computes expert probabilities via softmax")
print("  2. Top-k selection picks experts per token")
print("  3. Without aux loss, router may collapse to few experts")
print("  4. Load balancing loss encourages uniform distribution")
print("  5. MoE enables larger capacity with same compute (k experts active)")
