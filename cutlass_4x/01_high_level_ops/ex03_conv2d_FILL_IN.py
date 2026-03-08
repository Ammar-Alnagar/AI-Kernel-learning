"""
Module 01 — High-Level Operators
Exercise 03 — Conv2d with Implicit GEMM

LEVEL: 1 (High-level op API)

WHAT YOU'RE BUILDING:
  FP16 Conv2d using implicit GEMM algorithm — the core operation in CNN 
  backbones (ResNet, EfficientNet) and vision transformers with conv embeddings.
  This is the exact pattern used in NVIDIA TAO Toolkit and DeepStream pipelines.

OBJECTIVE:
  - Configure cutlass.op.Conv2d for standard convolution
  - Understand NHWC vs NCHW layout implications
  - Learn implicit GEMM: conv → im2col → GEMM fusion
  - Compare performance vs torch.nn.functional.conv2d
"""

import torch
import cutlass
import time
from dataclasses import dataclass
from typing import Tuple, Optional


# ==============================================================================
# PREDICT BEFORE RUNNING
# ==============================================================================
# Q1: What's the output shape for Conv2d with:
#     Input:  [N=32, C=64, H=224, W=224]
#     Weight: [K=128, C=64, R=3, S=3]  (K=output channels, R×S=kernel)
#     Padding: 1, Stride: 1
#     Expected output shape: ?

# Q2: Why does CUTLASS use NHWC layout instead of NCHW?
#     Hint: Consider memory coalescing for Tensor Core operations

# Q3: What is "implicit GEMM"? How does it differ from explicit im2col + GEMM?


# ==============================================================================
# SETUP
# ==============================================================================

@dataclass
class Conv2dConfig:
    N: int   # Batch size
    C: int   # Input channels
    H: int   # Input height
    W: int   # Input width
    K: int   # Output channels
    R: int   # Kernel height
    S: int   # Kernel width
    padding: Tuple[int, int]
    stride: Tuple[int, int]
    dilation: Tuple[int, int] = (1, 1)
    
    @property
    def output_h(self) -> int:
        """Compute output height."""
        return (self.H + 2 * self.padding[0] - self.dilation[0] * (self.R - 1) - 1) // self.stride[0] + 1
    
    @property
    def output_w(self) -> int:
        """Compute output width."""
        return (self.W + 2 * self.padding[1] - self.dilation[1] * (self.S - 1) - 1) // self.stride[1] + 1


# Standard ResNet-like configuration
config = Conv2dConfig(
    N=32, C=64, H=224, W=224,  # Input: 32 × 64 × 224 × 224
    K=128,                      # Output channels
    R=3, S=3,                   # 3×3 kernel
    padding=(1, 1),             # Same padding
    stride=(1, 1),
)

dtype = torch.float16
device = torch.device("cuda")

# Allocate tensors in NHWC layout (CUTLASS native)
# Note: PyTorch uses NCHW by default, we'll transpose for CUTLASS
input_nhwc = torch.randn(config.N, config.H, config.W, config.C, dtype=dtype, device=device)
weight_nhwc = torch.randn(config.R, config.S, config.C, config.K, dtype=dtype, device=device)
bias = torch.randn(config.K, dtype=dtype, device=device)

# Reference: PyTorch conv2d (NCHW)
input_nchw = input_nhwc.permute(0, 3, 1, 2).contiguous()  # NHWC → NCHW
weight_nchw = weight_nhwc.permute(3, 2, 0, 1).contiguous()  # NHWC → NCHW

def torch_conv2d_ref(input_nchw, weight_nchw, bias, config):
    """Reference Conv2d using PyTorch."""
    output = torch.nn.functional.conv2d(
        input_nchw, weight_nchw, bias=bias,
        stride=config.stride, padding=config.padding, dilation=config.dilation
    )
    return output.permute(0, 2, 3, 1).contiguous()  # NCHW → NHWC


C_ref = torch_conv2d_ref(input_nchw, weight_nchw, bias, config)

print(f"Conv2d Configuration:")
print(f"  Input:    [{config.N}, {config.H}, {config.W}, {config.C}] (NHWC)")
print(f"  Weight:   [{config.R}, {config.S}, {config.C}, {config.K}] (NHWC)")
print(f"  Output:   [{config.N}, {config.output_h}, {config.output_w}, {config.K}] (NHWC)")
print(f"  Padding:  {config.padding}, Stride: {config.stride}")


# ==============================================================================
# FILL IN: Level 1 — High-Level Op API for Conv2d
# ==============================================================================

print("\n" + "=" * 60)
print("LEVEL 1: High-Level Op API (cutlass.op.Conv2d)")
print("=" * 60)

# TODO [MEDIUM]: Configure Conv2d using cutlass.op.Conv2d
# HINT: Use cutlass.op.Conv2d with:
#   - element=cutlass.float16
#   - layout=cutlass.LayoutType.NHWC
#   - Specify problem size via Conv2dArguments
# REF: cutlass/examples/python/CuTeDSL/conv2d.py

# TODO: Import Conv2dArguments
# from cutlass.op.conv2d import Conv2dArguments

# TODO: Create Conv2d plan
# conv_plan = cutlass.op.Conv2d(
#     element=cutlass.float16,
#     layout=cutlass.LayoutType.NHWC,
# )

# TODO: Create arguments for Conv2d
# args = Conv2dArguments(
#     problem_size=(config.N, config.output_h, config.output_w, config.K),
#     filter_size=(config.R, config.S),
#     padding=config.padding,
#     stride=config.stride,
#     dilation=config.dilation,
# )

# TODO: Allocate output tensor
# C = torch.zeros(config.N, config.output_h, config.output_w, config.K, 
#                 dtype=dtype, device=device)

# TODO: Run Conv2d
# conv_plan.run(input_nhwc, weight_nhwc, C, args, bias=bias)

# Placeholder (replace with implementation)
C = torch.zeros(config.N, config.output_h, config.output_w, config.K, 
                dtype=dtype, device=device)

print(f"\nCUTLASS Conv2d completed")
print(f"Output shape: {C.shape}")


# ==============================================================================
# VERIFICATION
# ==============================================================================

# TODO [EASY]: Verify correctness against PyTorch reference
# HINT: Use torch.allclose with fp16 tolerances
# is_correct = torch.allclose(C, C_ref, rtol=..., atol=...)

is_correct = torch.allclose(C, C_ref, rtol=1e-2, atol=1e-2)
print(f"\nCorrectness check: {'✓ PASS' if is_correct else '✗ FAIL'}")

if not is_correct:
    max_error = (C - C_ref).abs().max().item()
    print(f"Max absolute error: {max_error:.6f}")


# ==============================================================================
# BENCHMARK
# ==============================================================================

def benchmark_cutlass_conv(plan, input tensor, weight, bias, args, C, 
                           num_warmup=10, num_iters=50):
    """Benchmark CUTLASS Conv2d latency."""
    # Warmup
    for _ in range(num_warmup):
        plan.run(input_tensor, weight, C, args, bias=bias)
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        plan.run(input_tensor, weight, C, args, bias=bias)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / num_iters) * 1000
    return avg_latency_ms


def benchmark_torch_conv(input_nchw, weight_nchw, bias, config, 
                         num_warmup=10, num_iters=50):
    """Benchmark PyTorch Conv2d latency."""
    # Warmup
    for _ in range(num_warmup):
        _ = torch.nn.functional.conv2d(
            input_nchw, weight_nchw, bias=bias,
            stride=config.stride, padding=config.padding
        )
    torch.cuda.synchronize()
    
    # Measure
    start = time.perf_counter()
    for _ in range(num_iters):
        output = torch.nn.functional.conv2d(
            input_nchw, weight_nchw, bias=bias,
            stride=config.stride, padding=config.padding
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    
    avg_latency_ms = (elapsed / num_iters) * 1000
    return avg_latency_ms


# TODO [MEDIUM]: Benchmark both implementations
# cutlass_latency = benchmark_cutlass_conv(...)
# torch_latency = benchmark_torch_conv(...)

cutlass_latency = 0.0
torch_latency = 0.0

print(f"\nPerformance:")
print(f"  CUTLASS Conv2d: {cutlass_latency:.3f} ms")
print(f"  PyTorch Conv2d: {torch_latency:.3f} ms")

if cutlass_latency > 0 and torch_latency > 0:
    speedup = torch_latency / cutlass_latency
    print(f"  Speedup:        {speedup:.2f}x")


# ==============================================================================
# COMPUTE TFLOPS
# ==============================================================================

def compute_conv_flops(config: Conv2dConfig) -> int:
    """
    Compute FLOPs for Conv2d.
    
    FLOPs = 2 × N × K × OH × OW × C × R × S
    (2 for multiply-add)
    """
    # TODO [EASY]: Implement FLOP calculation
    # Formula: 2 * N * K * output_h * output_w * C * R * S
    # flops = ...
    flops = 0
    return flops


flops = compute_conv_flops(config)
print(f"\nConvolution FLOPs: {flops / 1e9:.2f}G")

if cutlass_latency > 0:
    cutlass_tflops = flops / (cutlass_latency * 1e-3) / 1e12
    print(f"CUTLASS TFLOPS:    {cutlass_tflops:.1f}")

if torch_latency > 0:
    torch_tflops = flops / (torch_latency * 1e-3) / 1e12
    print(f"PyTorch TFLOPS:    {torch_tflops:.1f}")


# ==============================================================================
# CHECKPOINT
# ==============================================================================

print("\n" + "=" * 60)
print("CHECKPOINT")
print("=" * 60)

# C1: Predictions vs Reality
print("C1: Predictions vs Reality")
print(f"    Q1: Output shape?")
print(f"        Expected: [{config.N}, {config.output_h}, {config.output_w}, {config.K}]")
print(f"        Actual:   {C.shape}")

print("\n    Q2: Why NHWC?")
print("        Answer: NHWC provides contiguous memory access for Tensor Cores.")
print("                NCHW requires strided access on channel dimension,")
print("                reducing memory coalescing efficiency.")

print("\n    Q3: Implicit vs Explicit GEMM?")
print("        Implicit: Conv params passed to kernel, im2col done on-the-fly")
print("        Explicit: Separate im2col creates intermediate tensor (memory overhead)")
print("        Implicit saves memory and achieves better fusion.")

# C2: Profile with ncu
print("\nC2: Profile with ncu")
print("    Command:")
print(f"    ncu --metrics sm__inst_executed_pipe_tensor.sum,\\")
print(f"                l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum \\")
print(f"        python ex03_conv2d_FILL_IN.py")
print("\n    Look for:")
print("      - High tensor core utilization")
print("      - L1 cache hit rate (should be >80% for weight reuse)")

# C3: Job-relevant question
print("\nC3: Interview Question")
print("    Q: Why don't modern CNNs use explicit im2col + GEMM?")
print("    A: Memory overhead. For large images, im2col intermediate")
print("       tensor can be 10-100× the input size. Implicit GEMM")
print("       fuses im2col into the GEMM loop, loading and transforming")
print("       input tiles on-the-fly.")

print("\n    Q: When would you use NCHW vs NHWC?")
print("    A: Use NHWC for:")
print("       - GPU inference/training (Tensor Core optimization)")
print("       - CUTLASS, cuDNN, TensorRT")
print("       Use NCHW for:")
print("       - CPU operations (some libraries optimized for NCHW)")
print("       - Interoperability with PyTorch default")

# C4: Production guidance
print("\nC4: Production Conv2d Guidance")
print("    Framework          Layout    Backend")
print("    PyTorch (GPU)      NHWC      cuDNN (auto-converts)")
print("    TensorRT           NHWC      CUTLASS")
print("    DeepStream         NHWC      CUTLASS/cuDNN")
print("    TensorFlow         NHWC      cuDNN/MKLDNN")
print("\n    Rule: Use NHWC for GPU, let framework handle conversion.")

print("\n" + "=" * 60)
print("Exercise 03 Complete!")
print("=" * 60)
