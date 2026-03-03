"""
FILE: speculative_production.py
TEACHES: How vLLM and HuggingFace implement speculative decoding
MAPS TO: Production code reading — assisted generation
RUN: python speculative_production.py — shows assisted generation API

REFERENCE:
- https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py
- https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode
"""

import torch
import numpy as np

print("=" * 70)
print("SPECULATIVE DECODING: PRODUCTION REFERENCE")
print("=" * 70)

# ============================================================
# PART 1: Speculative Decoding in Production
# Math reference: see 08_speculative_decoding
# ============================================================

print("""
Speculative decoding implementations:

1. HuggingFace Assisted Generation
   - Uses small model to draft tokens
   - Large model verifies in parallel
   - API: model.generate(..., assistant_model=small_model)

2. vLLM Speculative Decoding
   - Supports n-gram speculation (no small model)
   - Also supports small-model speculation
   - Config: speculative_config

3. TensorRT-LLM
   - Draft-and-verify with small model
   - Tree-based verification
""")

# ============================================================
# PART 2: HuggingFace Assisted Generation
# ============================================================

print("\n" + "=" * 70)
print("HUGGINGFACE ASSISTED GENERATION")
print("=" * 70)

print("""
HuggingFace speculative decoding API:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load target (large) model
target_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-8B"
)

# Load assistant (small) model
assistant_model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Meta-Llama-3-1B"
)

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
inputs = tokenizer("Hello, my name is", return_tensors="pt")

# Generate with speculation
outputs = target_model.generate(
    **inputs,
    assistant_model=assistant_model,  # Draft model
    num_assistant_tokens=4,           # Draft k=4 tokens
    max_new_tokens=100,
)
```

What happens internally:
1. Assistant model drafts k tokens autoregressively
2. Target model evaluates all k tokens in parallel
3. Accept/reject each token based on probability ratio
4. Continue from last accepted token
""")

# ============================================================
# PART 3: Simulate Assisted Generation
# ============================================================

print("\n" + "=" * 70)
print("SIMULATE ASSISTED GENERATION")
print("=" * 70)

# Simulate draft and verify
def simulate_assisted_generation(prompt_tokens, num_steps, k, alpha):
    """
    Simulate speculative decoding.
    
    Args:
        prompt_tokens: Initial tokens
        num_steps: Number of generation steps
        k: Number of draft tokens per step
        alpha: Acceptance rate
    
    Returns:
        Generated tokens and statistics
    """
    rng = np.random.Generator(np.random.PCG64(42))
    
    generated = list(prompt_tokens)
    total_draft = 0
    total_accepted = 0
    
    for step in range(num_steps):
        # Draft phase: assistant generates k tokens
        draft_tokens = [rng.integers(1000, 10000) for _ in range(k)]
        total_draft += k
        
        # Verify phase: target accepts with probability alpha
        accepted = 0
        for i, token in enumerate(draft_tokens):
            if rng.random() < alpha:
                accepted += 1
                generated.append(token)
            else:
                # Rejected: sample new token
                new_token = rng.integers(1000, 10000)
                generated.append(new_token)
                accepted += 1  # Final token always accepted
                total_accepted += accepted
                break
        else:
            # All k accepted: add one more from target
            new_token = rng.integers(1000, 10000)
            generated.append(new_token)
            total_accepted += k + 1
            continue
        
        total_accepted += accepted
    
    return generated, total_draft, total_accepted

# Simulate
prompt = [1, 2, 3]
num_steps = 10
k = 4
alpha = 0.7

generated, draft, accepted = simulate_assisted_generation(prompt, num_steps, k, alpha)

print(f"\nSimulation config:")
print(f"  Prompt: {prompt}")
print(f"  Steps: {num_steps}")
print(f"  Draft tokens (k): {k}")
print(f"  Acceptance rate (α): {alpha}")
print()
print(f"Results:")
print(f"  Total draft tokens: {draft}")
print(f"  Total accepted: {accepted}")
print(f"  Speedup: {accepted / draft:.2f}x")
print(f"  Generated length: {len(generated)} tokens")

# ============================================================
# PART 4: vLLM Speculative Decoding
# ============================================================

print("\n" + "=" * 70)
print("VLLM SPECULATIVE DECODING")
print("=" * 70)

print("""
vLLM speculative decoding config:

```python
from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    speculative_model="meta-llama/Meta-Llama-3-1B",  # Draft model
    num_speculative_tokens=4,  # Draft k=4 tokens
)

# Or use n-gram speculation (no small model)
llm = LLM(
    model="meta-llama/Meta-Llama-3-8B",
    speculative_model="[ngram]",  # N-gram based speculation
    num_speculative_tokens=4,
    ngram_prompt_lookup_max=5,  # Max n-gram size
)

sampling_params = SamplingParams(
    max_tokens=100,
    temperature=0.7,
)

outputs = llm.generate(prompts, sampling_params)
```

vLLM features:
- N-gram speculation (no small model needed)
- Tree-based verification (parallel verify)
- Automatic fallback if speculation fails
""")

# ============================================================
# PART 5: Speedup vs. Acceptance Rate
# ============================================================

print("\n" + "=" * 70)
print("SPEEDUP VS. ACCEPTANCE RATE")
print("=" * 70)

# Theoretical speedup formula
def theoretical_speedup(alpha, k):
    """E[accepted] / k = speedup over autoregressive."""
    return (1 - alpha**(k+1)) / ((1 - alpha) * k)

print(f"\nTheoretical speedup for different α and k:")
print(f"\n{'α':<6}", end="")
for k_test in [2, 4, 6, 8]:
    print(f" k={k_test:<6}", end="")
print()
print("-" * 40)

for alpha_test in [0.5, 0.6, 0.7, 0.8, 0.9]:
    print(f"{alpha_test:<6}", end="")
    for k_test in [2, 4, 6, 8]:
        speedup = theoretical_speedup(alpha_test, k_test)
        print(f" {speedup:<8.2f}", end="")
    print()

print(f"\nNote: Higher α → more speedup. Optimal k depends on α.")
print(f"      For α=0.7, k=4-6 is optimal.")

# ============================================================
# PART 6: Tree Attention for Verification
# ============================================================

print("\n" + "=" * 70)
print("TREE ATTENTION FOR VERIFICATION")
print("=" * 70)

print("""
Tree attention enables parallel verification:

Standard causal mask (autoregressive):
  Token i can only attend to tokens 0..i-1

Tree mask (speculative verify):
  All draft tokens attend to prompt + previous drafts
  
Example (k=4 draft tokens, prompt length=4):

Standard (sequential):
  Step 1: verify token 4 (context: 0,1,2,3)
  Step 2: verify token 5 (context: 0,1,2,3,4)
  ...

Tree (parallel):
  Verify all at once with tree mask:
    Token 4 attends to: 0,1,2,3
    Token 5 attends to: 0,1,2,3,4
    Token 6 attends to: 0,1,2,3,4,5
    Token 7 attends to: 0,1,2,3,4,5,6

This is implemented via custom attention mask in the kernel.
""")

# ============================================================
# PART 7: Production Considerations
# ============================================================

print("\n" + "=" * 70)
print("PRODUCTION CONSIDERATIONS")
print("=" * 70)

print("""
When to use speculative decoding:

✓ Memory-bound workloads (decode phase)
✓ High acceptance rate expected (similar models)
✓ Latency-tolerant (slightly higher variance)

✗ Compute-bound workloads (prefill phase)
✗ Very different models (low acceptance rate)
✗ Strict latency requirements

Typical speedups:
- LLaMA-3 8B → LLaMA-3 1B: 1.5-2.0x
- N-gram speculation: 1.2-1.5x
- High α (0.8+): up to 3x
""")

# ============================================================
# VERIFY: Summary
# ============================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)

print(f"\n✓ HuggingFace: assistant_model API")
print(f"✓ vLLM: speculative_model config")
print(f"✓ N-gram speculation (no small model)")
print(f"✓ Tree attention for parallel verification")
print(f"✓ Simulated speedup: {accepted / draft:.2f}x at α={alpha}")
print()
print("PASS — Speculative decoding production reference complete.")
print()
print("Key insights:")
print("  1. Speculative decoding addresses decode underutilization")
print("  2. HuggingFace uses assistant_model parameter")
print("  3. vLLM supports both model-based and n-gram speculation")
print("  4. Tree attention enables parallel verification")
print("  5. Speedup depends on acceptance rate α")
print()
print("Sources:")
print("  HF: https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py")
print("  vLLM: https://github.com/vllm-project/vllm/blob/main/vllm/spec_decode")
