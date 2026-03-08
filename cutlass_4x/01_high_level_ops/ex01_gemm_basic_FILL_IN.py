"""
Module 01 — High-Level Operators
Exercise 01 — Basic GEMM with cutlass.op

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  FP16 GEMM (General Matrix Multiply) — the foundational operation for all 
  deep learning workloads. This is the exact pattern used in transformer 
  attention (QKV projections) and MLP layers.

OBJECTIVE:
  - Configure and run a basic GEMM using cutlass.op.Gemm
  - Understand tensor layout requirements (row-major vs column-major)
  - Verify correctness against torch.mm reference
  - Measure achieved TFLOPS vs theoretical peak
"""

import torch
import cutlass
from cutlass.cute.runtime import from_dlpack
import time


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What output shape do you expect from A @ B where A=[M, K] and B=[K, N]?
#     A.shape = [512, 1024], B.shape = [1024, 2048]
#     Expected C.shape = ?

# Q2: Will CUTLASS GEMM be faster or slower than torch.mm for this matrix size?
#     Why? Consider kernel launch overhead vs compute time.

# Q3: What's the arithmetic intensity (ops/byte) for this GEMM?
#     Formula: (2 * M * N * K) / (M*K + K*N + M*N) elements
#     Hint: GEMM is compute-bound when arithmetic intensity > 10 ops/byte


# ==============================================================================
# SETUP
# ==============================================================================

# Matrix dimensions
M = 512   # Batch / sequence dimension
K = 1024  # Input / hidden dimension  
N = 2048  # Output dimension

# Data type
dtype = torch.float16

# Allocate tensors on CUDA
device = torch.device("cuda")
A = torch.randn(M, K, dtype=dtype, device=device)
B = torch.randn(K, N, dtype=dtype, device=device)
C = torch.zeros(M, N, dtype=dtype, device=device)

# Reference output (torch.mm)
C_ref = torch.mm(A, B)


# ==============================================================================
# FILL IN: Level 1 — High-Level Op API
# ==============================================================================

print("=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op)")
print("=" * 60)

# TODO [EASY]: Configure the GEMM plan
# HINT: Use cutlass.op.Gemm with element=cutlass.float16
#       and layout=cutlass.LayoutType.RowMajor
# REF: cutlass/examples/python/CuTeDSL/gemm_basic.py

# Create the GEMM plan
# plan = cutlass.op.Gemm(...)

# TODO [EASY]: Run the GEMM operation
# HINT: plan.run(A, B, C) computes C = A @ B
# plan.run(...)

print(f"\nCUTLASS GEMM completed")
print(f"Output C.shape = {C.shape}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

# TODO [EASY]: Verify correctness against torch.mm reference
# HINT: Use torch.allclose with appropriate tolerances for FP16
#       fp16 has ~3 decimal digits of precision, so rtol=1e-3, atol=1e-3
# is_correct = torch.allclose(...)

is_correct = torch.allclose(C, C_ref, rtol=1e-3, atol=1e-3)
print(f"\nCorrectness check: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (C - C_ref).abs().max().item()
    print(f"Max absolute error: {max_error:.6f}")


# ==============================================================================
# BENCHMARK
# ==============================================================================

def benchmark_gemm(plan, A, B, C, num_warmup=10, num_iters=100):
    """Benchmark GEMM latency and compute TFLOPS."""
    # Warmup
    for _ in range(num_warmup):
        plan.run(A, B, C)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        plan.run(A, B, C)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / num_iters) * 1000
    
    # Compute TFLOPS
    # GEMM FLOPs = 2 * M * N * K (multiply-add counted as 2 ops)
    flops = 2 * M * N * K
    tflops = flops / (avg_latency_ms * 1e-3) / 1e12
    
    return avg_latency_ms, tflops


# TODO [MEDIUM]: Benchmark the CUTLASS GEMM
# HINT: Call benchmark_gemm(plan, A, B, C)
# avg_latency, tflops = benchmark_gemm(...)

avg_latency, tflops = benchmark_gemm(plan, A, B, C)

print(f"\nPerformance:")
print(f"  Average latency: {avg_latency:.3f} ms")
print(f"  Achieved TFLOPS: {tflops:.1f}")


# ==============================================================================
# COMPARE WITH TORCH
# ==============================================================================

def benchmark_torch_mm(A, B, C, num_warmup=10, num_iters=100):
    """Benchmark torch.mm latency."""
    for _ in range(num_warmup):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    
    start = time.perf_counter()
    for _ in range(num_iters):
        C.copy_(torch.mm(A, B))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / num_iters) * 1000
    flops = 2 * M * N * K
    tflops = flops / (avg_latency_ms * 1e-3) / 1e12
    
    return avg_latency_ms, tflops


torch_latency, torch_tflops = benchmark_torch_mm(A, B, C)

print(f"\nComparison with torch.mm:")
print(f"  torch.mm latency:  {torch_latency:.3f} ms ({torch_tflops:.1f} TFLOPS)")
print(f"  CUTLASS latency:   {avg_latency:.3f} ms ({tflops:.1f} TFLOPS)")

speedup = torch_latency / avg_latency
print(f"  Speedup:           {speedup:.2f}x")


# ==============================================================================
# CHECKPOINT
# ==============================================================================
print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Did predictions match?
print("C1: Predictions vs Reality")
print(f"    - Expected C.shape: [M, N] = [{M}, {N}]")
print(f"    - Actual C.shape:   {C.shape}")
print(f"    - torch.mm faster/slower? {'torch.mm' if torch_latency < avg_latency else 'CUTLASS'} won")

# C2: Profile with ncu
print("\nC2: Profile with ncu to identify bottlenecks")
print("    Command:")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum,l2tex__t_bytes.sum \\")
print(f"        python ex01_gemm_basic_FILL_IN.py")
print("\n    Look for:")
print("      - High tensor core utilization (sm__inst_executed_pipe_tensor)")
print("      - Memory bandwidth vs compute bound (compare l2tex__t_bytes to tensor ops)")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: Why might CUTLASS be slower than torch.mm for small matrices?")
print("    A: Kernel launch overhead dominates for small workloads.")
print("       torch.mm uses highly optimized cuBLAS with persistent kernels.")
print("       CUTLASS shines when you need custom fusion or non-standard layouts.")

# C4: Arithmetic intensity calculation
print("\nC4: Arithmetic Intensity Calculation")
num_elements_input = M * K + K * N  # A + B
num_elements_output = M * N          # C
bytes_per_element = 2                # FP16 = 2 bytes
total_bytes = (num_elements_input + num_elements_output) * bytes_per_element
total_ops = 2 * M * N * K            # 2 ops per FMA
arithmetic_intensity = total_ops / total_bytes
print(f"    Total ops:        {total_ops:,} ({total_ops/1e9:.1f}G ops)")
print(f"    Total bytes:      {total_bytes:,} ({total_bytes/1e6:.1f}MB)")
print(f"    Arithmetic Int.:  {arithmetic_intensity:.2f} ops/byte")
print(f"    Classification:   {'Compute-bound' if arithmetic_intensity > 10 else 'Memory-bound'}")

print("\n" + "=" * 60)
print("Exercise 01 Complete!")
print("=" * 60)
