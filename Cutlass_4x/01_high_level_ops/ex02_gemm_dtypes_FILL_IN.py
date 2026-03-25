"""
Module 01 — High-Level Operators
Exercise 02 — Mixed Precision GEMM (FP16, BF16, FP8, TF32)

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  Mixed-precision GEMM configurations — the foundation of modern inference 
  optimization. FP8 is used in TensorRT-LLM for 2× memory bandwidth savings.
  BF16 is the training standard for LLMs (better dynamic range than FP16).

OBJECTIVE:
  - Configure GEMM for FP16, BF16, FP8 (E4M3/E5M2), and TF32
  - Understand accumulator type requirements (FP32 for low-precision inputs)
  - Compare performance and accuracy across dtypes
  - Learn when to use each precision in production
"""

import torch
import cutlass
from dataclasses import dataclass
from typing import Tuple
import time


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: Which dtype will be fastest: FP16, BF16, or FP8?
#     Consider: memory bandwidth vs tensor core throughput

# Q2: What's the maximum relative error you expect for FP8 E4M3 vs FP32 reference?
#     Hint: FP8 E4M3 has ~3 bits of mantissa = ~1 decimal digit precision

# Q3: Why does TF32 exist? What problem does it solve vs FP16?


# ==============================================================================
# SETUP
# ==============================================================================

@dataclass
class DtypeConfig:
    name: str
    element_type: cutlass.DataType
    torch_dtype: torch.dtype
    accum_type: cutlass.DataType
    torch_accum_dtype: torch.dtype
    description: str


# Define configurations for different dtypes
DTYPE_CONFIGS = {
    "fp16": DtypeConfig(
        name="FP16",
        element_type=cutlass.float16,
        torch_dtype=torch.float16,
        accum_type=cutlass.float32,
        torch_accum_dtype=torch.float32,
        description="Standard mixed-precision training (Ampere+)"
    ),
    "bf16": DtypeConfig(
        name="BF16",
        element_type=cutlass.bfloat16,
        torch_dtype=torch.bfloat16,
        accum_type=cutlass.float32,
        torch_accum_dtype=torch.float32,
        description="LLM training standard (better range than FP16)"
    ),
    "tf32": DtypeConfig(
        name="TF32",
        element_type=cutlass.tf32,
        torch_dtype=torch.float32,  # TF32 uses FP32 tensors but lower precision
        torch_dtype=torch.float32,
        accum_type=cutlass.float32,
        torch_accum_dtype=torch.float32,
        description="TensorFloat32 (Ampere+ default for FP32 GEMM)"
    ),
    # FP8 requires Hopper (SM90) or later
    "fp8_e4m3": DtypeConfig(
        name="FP8 E4M3",
        element_type=cutlass.float8_e4m3fn,
        torch_dtype=torch.float8_e4m3fn,
        accum_type=cutlass.float32,
        torch_accum_dtype=torch.float32,
        description="FP8 inference (E4M3 for weights/activations)"
    ),
    "fp8_e5m2": DtypeConfig(
        name="FP8 E5M2",
        element_type=cutlass.float8_e5m2,
        torch_dtype=torch.float8_e5m2,
        accum_type=cutlass.float32,
        torch_accum_dtype=torch.float32,
        description="FP8 with extended range (E5M2 for gradients)"
    ),
}

# Matrix dimensions
M, K, N = 1024, 2048, 1024

# Reference computation in FP32
device = torch.device("cuda")
A_fp32 = torch.randn(M, K, dtype=torch.float32, device=device)
B_fp32 = torch.randn(K, N, dtype=torch.float32, device=device)
C_ref = torch.mm(A_fp32, B_fp32)


# ==============================================================================
# FILL IN: Level 1 — High-Level Op API with Different Dtypes
# ==============================================================================

def run_gemm_with_dtype(config: DtypeConfig) -> Tuple[torch.Tensor, float]:
    """
    Run GEMM with specified dtype and return result + latency.
    
    TODO [MEDIUM]: Implement mixed-precision GEMM
    HINT: 
      1. Convert inputs to target dtype
      2. Configure cutlass.op.Gemm with element=config.element_type
      3. Set accumulator_type=config.accum_type for proper precision
      4. Run and measure latency
    REF: cutlass/examples/python/CuTeDSL/gemm_mixed_precision.py
    """
    
    # TODO: Convert inputs to target dtype
    # A = A_fp32.to(config.torch_dtype)
    # B = B_fp32.to(config.torch_dtype)
    
    # TODO: Allocate output tensor in accumulator dtype
    # C = torch.zeros(M, N, dtype=config.torch_accum_dtype, device=device)
    
    # TODO: Configure GEMM plan
    # plan = cutlass.op.Gemm(
    #     element=...,
    #     accumulator_type=...,
    #     layout=cutlass.LayoutType.RowMajor
    # )
    
    # TODO: Run GEMM and measure latency
    # start = time.perf_counter()
    # plan.run(A, B, C)
    # torch.cuda.synchronize()
    # latency = time.perf_counter() - start
    
    # Placeholder (replace with implementation)
    C = torch.zeros(M, N, dtype=config.torch_accum_dtype, device=device)
    latency = 0.0
    
    return C, latency * 1000  # Return latency in ms


print("=" * 60)
print("Mixed Precision GEMM Benchmark")
print("=" * 60)

results = {}

for dtype_key, config in DTYPE_CONFIGS.items():
    print(f"\n{config.name}: {config.description}")
    print(f"  Element: {config.element_type}, Accumulator: {config.accum_type}")
    
    try:
        C_result, latency_ms = run_gemm_with_dtype(config)
        
        # Convert result to FP32 for comparison
        C_fp32 = C_result.to(torch.float32)
        
        # Compute error vs FP32 reference
        max_abs_error = (C_fp32 - C_ref).abs().max().item()
        max_rel_error = (C_fp32 - C_ref).abs().div(C_ref.abs() + 1e-8).max().item()
        
        # Compute TFLOPS
        flops = 2 * M * N * K
        tflops = flops / (latency_ms * 1e-3) / 1e12 if latency_ms > 0 else 0
        
        results[dtype_key] = {
            "latency_ms": latency_ms,
            "tflops": tflops,
            "max_abs_error": max_abs_error,
            "max_rel_error": max_rel_error,
        }
        
        print(f"  Latency: {latency_ms:.3f} ms, TFLOPS: {tflops:.1f}")
        print(f"  Max Abs Error: {max_abs_error:.6f}, Max Rel Error: {max_rel_error:.4f}")
        
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        results[dtype_key] = None


# ==============================================================================
# ANALYSIS
# ==============================================================================

print("\n" + "=" * 60)
print("Analysis")
print("=" * 60)

# Find fastest dtype
valid_results = {k: v for k, v in results.items() if v is not None}
if valid_results:
    fastest = min(valid_results.items(), key=lambda x: x[1]["latency_ms"])
    print(f"\nFastest dtype: {fastest[0].upper()} ({fastest[1]['latency_ms']:.3f} ms)")
    
    # Find most accurate
    most_accurate = min(valid_results.items(), key=lambda x: x[1]["max_rel_error"])
    print(f"Most accurate: {most_accurate[0].upper()} (rel error: {most_accurate[1]['max_rel_error']:.6f})")
    
    # Print comparison table
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print(f"{'Dtype':<12} {'Latency (ms)':<14} {'TFLOPS':<10} {'Rel Error':<12}")
    print("-" * 60)
    for dtype_key, res in valid_results.items():
        print(f"{dtype_key.upper():<12} {res['latency_ms']:<14.3f} {res['tflops']:<10.1f} {res['max_rel_error']:<12.6f}")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print("    Q1: Which dtype was fastest?")
if valid_results:
    fastest_key = min(valid_results.items(), key=lambda x: x[1]["latency_ms"])[0]
    print(f"        Answer: {fastest_key.upper()}")
print("\n    Q2: FP8 E4M3 relative error?")
if "fp8_e4m3" in valid_results:
    print(f"        Actual: {valid_results['fp8_e4m3']['max_rel_error']:.4f}")
print("\n    Q3: Why TF32?")
print("        Answer: TF32 provides FP32 convenience with ~7 digits precision")
print("                (vs FP16's ~3 digits) while maintaining 2× FP32 throughput")
print("                on Ampere+ Tensor Cores. Default for torch.matmul on A100+.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command for FP8:")
print("    ncu --metrics sm__inst_executed_pipe_tensor.sum,\\")
print("                l2tex__t_bytes.sum,\\")
print("                dram__throughput.sum \\")
print("        python ex02_gemm_dtypes_FILL_IN.py")
print("\n    Look for:")
print("      - FP8 should show 2× memory bandwidth efficiency vs FP16")
print("      - Tensor core utilization should be similar across dtypes")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: When would you choose BF16 over FP16 for LLM training?")
print("    A: BF16 has 8-bit exponent (vs FP16's 5-bit), giving FP32-like")
print("       dynamic range. Critical for:")
print("       - Large batch training (gradient accumulation)")
print("       - Deep networks (vanishing/exploding gradients)")
print("       - Mixed precision without loss scaling complexity")
print("\n    Q: When is FP8 appropriate?")
print("    A: FP8 inference when:")
print("       - 2× memory bandwidth savings matter (large models)")
print("       - ~1% accuracy degradation is acceptable")
print("       - Hardware supports it natively (Hopper+, RTX 4090)")

# C4: Production guidance
print("\nC4: Production Dtype Selection Guide")
print("    Training:")
print("      - LLM pretraining:        BF16 (stability)")
print("      - Fine-tuning:            FP16 or BF16 (model-dependent)")
print("      - Vision models:          FP16 (sufficient range)")
print("\n    Inference:")
print("      - High accuracy:          FP16")
print("      - Throughput-optimized:   FP8 E4M3 (weights + activations)")
print("      - Legacy hardware:        TF32 (Ampere default)")

print("\n" + "=" * 60)
print("Exercise 02 Complete!")
print("=" * 60)
