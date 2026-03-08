"""
Module 08 — Timing and Benchmarking
Exercise 01 — CUDA Events Timing

WHAT YOU'RE BUILDING:
  Accurate kernel timing requires CUDA events, not Python time.time().
  CUDA events measure GPU time, excluding CPU launch overhead.
  This is how all serious kernel benchmarks work.

OBJECTIVE:
  - Use torch.cuda.Event for GPU timing
  - Understand sync requirements for accurate timing
  - Calculate ms/iter and TFLOPS from raw timing
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: Why can't you use time.time() for accurate GPU kernel timing?
# Q2: What does torch.cuda.synchronize() do and when do you need it?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from typing import Tuple

CUDA_AVAILABLE = torch.cuda.is_available()

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Time a kernel using CUDA events.
#              This is the standard pattern for GPU benchmarking.
#              1. Create start/end events
#              2. Record start, run kernel, record end
#              3. Synchronize and compute elapsed time
# HINT: start.record(); kernel(); end.record(); torch.cuda.synchronize(); start.elapsed_time(end)

def time_kernel_cuda(
    kernel_fn,
    *args,
    num_iters: int = 10
) -> Tuple[float, float]:
    """Time a kernel using CUDA events.
    
    Args:
        kernel_fn: Function to time
        *args: Arguments to pass to kernel_fn
        num_iters: Number of iterations for averaging
    
    Returns:
        (mean_ms, std_ms) per iteration
    """
    if not CUDA_AVAILABLE:
        print("CUDA not available, returning dummy values")
        return (1.0, 0.1)
    
    # TODO: implement CUDA event timing
    # 1. Create events
    # 2. Warmup run
    # 3. Record multiple iterations
    # 4. Compute mean and std
    pass

# TODO [EASY]: Calculate TFLOPS from timing and problem size.
#              For matmul: FLOPS = 2 * M * N * K
# HINT: tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12

def calculate_tflops(M: int, N: int, K: int, ms: float) -> float:
    """Calculate TFLOPS for matmul.
    
    Args:
        M, N, K: Matrix dimensions
        ms: Time in milliseconds
    
    Returns:
        TFLOPS (tera floating-point operations per second)
    """
    # TODO: implement TFLOPS calculation
    pass

# TODO [MEDIUM]: Full benchmark function that returns timing + TFLOPS.
#              This is the pattern you'll use in the capstone.
# HINT: Combine time_kernel_cuda and calculate_tflops

def benchmark_matmul(
    M: int, N: int, K: int,
    kernel_fn=None,
    num_iters: int = 10
) -> dict:
    """Benchmark a matmul kernel.
    
    Args:
        M, N, K: Matrix dimensions
        kernel_fn: Optional custom kernel (default: torch.matmul)
        num_iters: Number of iterations
    
    Returns:
        Dict with keys: ms_mean, ms_std, tflops, M, N, K
    """
    if kernel_fn is None:
        kernel_fn = lambda a, b: torch.matmul(a, b)
    
    # TODO: implement full benchmark
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why do you need multiple iterations for benchmarking?
# C2: What's the theoretical peak TFLOPS of your GPU?

if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("CUDA not available, skipping benchmark")
    else:
        print("Testing CUDA event timing...\n")
        
        M, N, K = 1024, 1024, 1024
        
        result = benchmark_matmul(M, N, K, num_iters=20)
        print(f"Benchmark result for {M}x{N}x{K}:")
        print(f"  Mean time: {result['ms_mean']:.3f} ms")
        print(f"  Std time: {result['ms_std']:.3f} ms")
        print(f"  TFLOPS: {result['tflops']:.1f}\n")
    
    print("Done!")
