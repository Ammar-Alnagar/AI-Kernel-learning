"""
Module 01 — High-Level Operators
Exercise 04 — Grouped GEMM for MoE (Mixture of Experts)

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  Variable-batch Grouped GEMM — the exact pattern used in MoE (Mixture of 
  Experts) layers for LLMs like Mixtral, Grok, and GShard. This enables 
  efficient expert routing where different tokens go to different experts.

OBJECTIVE:
  - Configure cutlass.op.GroupedGemm for variable-batch GEMM
  - Understand MoE routing and expert parallelism
  - Compare grouped GEMM vs naive loop over experts
  - Measure tokens/sec improvement
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import List, Tuple


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: Why is Grouped GEMM faster than looping over experts?
#     Hint: Consider kernel launch overhead and memory access patterns

# Q2: In MoE, if you have 8 experts and batch_size=1024 tokens, with 
#     top-2 routing (each token goes to 2 experts), what's the average
#     batch size per expert?

# Q3: What's the memory layout requirement for grouped GEMM inputs?
#     Can all GEMMs in the group have different shapes?


# ==============================================================================
# SETUP
# ==============================================================================

@dataclass
class GemmProblem:
    """Descriptor for a single GEMM in the group."""
    M: int  # Batch dimension (tokens for this expert)
    K: int  # Input dimension (hidden size)
    N: int  # Output dimension (expert width)


# MoE Configuration
NUM_EXPERTS = 8
HIDDEN_SIZE = 1024
EXPERT_WIDTH = 2048
TOTAL_TOKENS = 4096
TOP_K = 2  # Each token routed to top-2 experts

dtype = torch.float16
device = torch.device("cuda")

# Simulate MoE routing: assign tokens to experts
# In real MoE, this comes from router network (softmax over expert scores)
torch.manual_seed(42)
token_expert_assignment = torch.randint(0, NUM_EXPERTS, (TOTAL_TOKENS, TOP_K), device=device)

# Count tokens per expert
tokens_per_expert = torch.zeros(NUM_EXPERTS, dtype=torch.int32, device=device)
for i in range(NUM_EXPERTS):
    tokens_per_expert[i] = (token_expert_assignment == i).sum()

print(f"MoE Configuration:")
print(f"  Total tokens:     {TOTAL_TOKENS}")
print(f"  Num experts:      {NUM_EXPERTS}")
print(f"  Hidden size:      {HIDDEN_SIZE}")
print(f"  Expert width:     {EXPERT_WIDTH}")
print(f"  Top-K routing:    {TOP_K}")
print(f"\nTokens per expert:")
for i, count in enumerate(tokens_per_expert):
    print(f"  Expert {i}: {count.item():4d} tokens")

# Create problem descriptors for each expert
problems: List[GemmProblem] = []
for i in range(NUM_EXPERTS):
    M = tokens_per_expert[i].item()
    K = HIDDEN_SIZE
    N = EXPERT_WIDTH
    problems.append(GemmProblem(M=M, K=K, N=N))

# Allocate input tensors (one per expert, with different M)
# In practice, you'd gather tokens by expert assignment
As = []
Bs = []
Cs = []
for prob in problems:
    A = torch.randn(prob.M, prob.K, dtype=dtype, device=device)
    B = torch.randn(prob.K, prob.N, dtype=dtype, device=device)
    C = torch.zeros(prob.M, prob.N, dtype=dtype, device=device)
    As.append(A)
    Bs.append(B)
    Cs.append(C)

# Reference: naive loop over experts
def naive_moe_gemm(As, Bs, Cs):
    """Naive MoE: loop over experts with separate GEMMs."""
    for i in range(len(As)):
        Cs[i].copy_(torch.mm(As[i], Bs[i]))
    return Cs


print("\n" + "=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op.GroupedGemm)")
print("=" * 60)

# TODO [HARD]: Configure Grouped GEMM using cutlass.op.GroupedGemm
# HINT: 
#   - Use cutlass.op.GroupedGemm with element=cutlass.float16
#   - Pass list of problem shapes
#   - Use plan.run() with lists of tensors
# REF: cutlass/examples/python/CuTeDSL/grouped_gemm.py

# TODO: Create GroupedGemm plan
# grouped_plan = cutlass.op.GroupedGemm(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.RowMajor
# )

# TODO: Run grouped GEMM
# grouped_plan.run(As, Bs, Cs, problems)

# Placeholder (replace with implementation)
grouped_plan = None

print(f"\nGrouped GEMM configuration complete")
print(f"Number of GEMMs in group: {len(problems)}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

# Run naive reference
Cs_ref = naive_moe_gemm(As, Bs, [torch.zeros_like(C) for C in Cs])

# Run grouped GEMM (if implemented)
if grouped_plan is not None:
    # Reset outputs
    Cs_grouped = [torch.zeros_like(C) for C in Cs]
    grouped_plan.run(As, Bs, Cs_grouped, problems)
    
    # Verify correctness
    all_correct = True
    for i, (C_ref, C_grp) in enumerate(zip(Cs_ref, Cs_grouped)):
        if not torch.allclose(C_ref, C_grp, rtol=1e-2, atol=1e-2):
            print(f"Expert {i}: ✗ FAIL")
            all_correct = False
    
    print(f"\nCorrectness check: {'✓ PASS' if all_correct else '✗ FAIL'}")
else:
    print("\nSkipping verification (grouped GEMM not implemented yet)")
    Cs_grouped = Cs_ref


# ==============================================================================
# BENCHMARK: Naive vs Grouped
# ==============================================================================

def benchmark_naive_moe(As, Bs, Cs, num_warmup=10, num_iters=50):
    """Benchmark naive MoE (loop over experts)."""
    # Warmup
    for _ in range(num_warmup):
        naive_moe_gemm(As, Bs, Cs)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        naive_moe_gemm(As, Bs, Cs)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / num_iters) * 1000
    return avg_latency_ms


def benchmark_grouped_moe(plan, As, Bs, Cs, problems, num_warmup=10, num_iters=50):
    """Benchmark grouped MoE."""
    if plan is None:
        return 0.0
    
    # Warmup
    for _ in range(num_warmup):
        plan.run(As, Bs, Cs, problems)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        plan.run(As, Bs, Cs, problems)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / num_iters) * 1000
    return avg_latency_ms


# TODO [MEDIUM]: Benchmark both approaches
# naive_latency = benchmark_naive_moe(As, Bs, Cs_ref)
# grouped_latency = benchmark_grouped_moe(grouped_plan, As, Bs, Cs_grouped, problems)

naive_latency = 0.0
grouped_latency = 0.0

print(f"\nPerformance:")
print(f"  Naive MoE (loop):    {naive_latency:.3f} ms")
print(f"  Grouped MoE:         {grouped_latency:.3f} ms")

if naive_latency > 0 and grouped_latency > 0:
    speedup = naive_latency / grouped_latency
    print(f"  Speedup:               {speedup:.2f}x")
    
    # Compute tokens/sec
    total_tokens_processed = TOTAL_TOKENS * TOP_K  # Each token processed by 2 experts
    tokens_per_sec_naive = total_tokens_processed / (naive_latency * 1e-3)
    tokens_per_sec_grouped = total_tokens_processed / (grouped_latency * 1e-3)
    
    print(f"\nThroughput:")
    print(f"  Naive MoE:    {tokens_per_sec_naive/1e6:.2f}M tokens/sec")
    print(f"  Grouped MoE:  {tokens_per_sec_grouped/1e6:.2f}M tokens/sec")


# ==============================================================================
# COMPUTE TFLOPS
# ==============================================================================

def compute_grouped_gemm_flops(problems: List[GemmProblem]) -> int:
    """Compute total FLOPs for grouped GEMM."""
    # TODO [EASY]: Sum FLOPs across all GEMMs
    # Formula per GEMM: 2 * M * N * K
    # total_flops = sum(...)
    total_flops = 0
    return total_flops


total_flops = compute_grouped_gemm_flops(problems)
print(f"\nTotal FLOPs: {total_flops / 1e9:.2f}G")

if grouped_latency > 0:
    grouped_tflops = total_flops / (grouped_latency * 1e-3) / 1e12
    print(f"Grouped TFLOPS: {grouped_tflops:.1f}")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Why is Grouped GEMM faster?")
print("        Answer: Single kernel launch handles all experts,")
print("                reducing launch overhead. Also enables better")
print("                GPU utilization by keeping all SMs busy with")
print("                work from multiple experts.")

print("\n    Q2: Average tokens per expert?")
avg_tokens = (TOTAL_TOKENS * TOP_K) / NUM_EXPERTS
print(f"        Calculation: ({TOTAL_TOKENS} tokens × {TOP_K}) / {NUM_EXPERTS} experts")
print(f"        Answer: {avg_tokens:.0f} tokens/expert (on average)")
print(f"                Actual distribution: {tokens_per_expert.tolist()}")

print("\n    Q3: Can GEMMs have different shapes?")
print("        Answer: Yes! That's the key feature of Grouped GEMM.")
print("                Each expert can have different M (tokens assigned).")
print("                K and N are typically the same, but can differ.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum,\\")
print(f"                dram__throughput.sum \\")
print(f"        python ex04_grouped_gemm_FILL_IN.py")
print("\n    Look for:")
print("      - Higher SM utilization for grouped vs naive")
print("      - Better memory coalescing (fewer, larger transfers)")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: How does MoE routing work in Mixtral/Grok?")
print("    A: Router network (linear layer + softmax) produces expert")
print("       scores for each token. Top-K experts selected. Tokens")
print("       are gathered by expert, processed in parallel via")
print("       Grouped GEMM, then scattered back to original positions.")

print("\n    Q: What's the load balancing challenge in MoE?")
print("    A: If routing is imbalanced (some experts overloaded),")
print("       you get stragglers. Solutions:")
print("       - Auxiliary load balancing loss")
print("       - Capacity factor (drop tokens if expert full)")
print("       - Dynamic expert assignment")

# C4: Production guidance
print("\nC4: Production MoE Implementation Tips")
print("    1. Use Grouped GEMM for expert processing (2-5× speedup)")
print("    2. Fuse router + top-K selection into single kernel")
print("    3. Use FP8 for expert weights (2× memory savings)")
print("    4. Consider expert parallelism across GPUs for large MoE")
print("    5. Profile tokens/sec, not just latency (variable batch)")

print("\n" + "=" * 60)
print("Exercise 04 Complete!")
print("=" * 60)
