"""
Module 08 — Timing and Benchmarking
Exercise 03 — Benchmark Harness

WHAT YOU'RE BUILDING:
  A complete benchmark harness ties together everything: CLI args,
  CUDA timing, result collection, and formatted output. This is
  the foundation for the capstone.

OBJECTIVE:
  - Combine all previous concepts into a working harness
  - Format output as a table
  - Save results to CSV
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What components does a complete benchmark harness need?
# Q2: How do you compute "vs baseline" for each result?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
import time
from dataclasses import dataclass
from typing import List, Dict, Optional
from pathlib import Path

CUDA_AVAILABLE = torch.cuda.is_available()

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    size: int
    ms: float
    tflops: float
    baseline_ratio: float = 1.0

# TODO [EASY]: Run a single benchmark iteration.
#              Use CUDA events for timing.
# HINT: Same pattern as ex01

def run_single_benchmark(size: int, num_iters: int = 10) -> float:
    """Run benchmark for a single size.
    
    Args:
        size: Matrix dimension (size x size x size)
        num_iters: Number of iterations
    
    Returns:
        Mean time in ms
    """
    if not CUDA_AVAILABLE:
        return 1.0
    
    # TODO: implement CUDA event timing for matmul
    pass

# TODO [MEDIUM]: Calculate TFLOPS for square matmul.
#              FLOPS = 2 * N^3 for N x N x N matmul
# HINT: tflops = (2 * size**3) / (ms * 1e-3) / 1e12

def calculate_tflops(size: int, ms: float) -> float:
    """Calculate TFLOPS for square matmul."""
    # TODO: implement
    pass

# TODO [MEDIUM]: Run benchmarks for multiple sizes.
#              Collect results with dict/list comprehension.
# HINT: [BenchmarkResult(...) for size in sizes]

def run_all_benchmarks(
    sizes: List[int],
    num_iters: int = 10
) -> List[BenchmarkResult]:
    """Run benchmarks for all sizes.
    
    Args:
        sizes: List of matrix dimensions
        num_iters: Iterations per benchmark
    
    Returns:
        List of BenchmarkResult
    """
    results = []
    baseline_ms = None
    
    for size in sizes:
        # TODO: run benchmark, calculate tflops, baseline ratio
        pass
    
    return results

# TODO [EASY]: Print results as a formatted table.
#              Use f-strings with field width.
# HINT: f"{value:>10.2f}" for right-aligned, 10 chars, 2 decimals

def print_results_table(results: List[BenchmarkResult]):
    """Print results as formatted table."""
    # TODO: implement table printing
    # Format:
    # ┌────────┬──────────┬───────────┬──────────────┐
    # │  Size  │  Ms/iter │  TFLOPS   │  vs baseline │
    # ├────────┼──────────┼───────────┼──────────────┤
    # │  1024  │   0.21   │  102.4    │    1.0×      │
    # └────────┴──────────┴───────────┴──────────────┘
    pass

# TODO [EASY]: Save results to CSV.
#              Use the pathlib pattern from Module 07.
# HINT: Same as ex02 in Module 07

def save_results_csv(results: List[BenchmarkResult], output_path: str):
    """Save results to CSV file."""
    # TODO: implement
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How does TFLOPS scale with matrix size for matmul?
# C2: Why is the first result used as baseline?

if __name__ == "__main__":
    if not CUDA_AVAILABLE:
        print("CUDA not available, skipping benchmark")
    else:
        print("Running benchmark harness...\n")
        
        sizes = [512, 1024, 2048]
        results = run_all_benchmarks(sizes, num_iters=10)
        
        print_results_table(results)
        
        save_results_csv(results, "benchmark_results.csv")
        print("\nSaved to benchmark_results.csv")
    
    print("\nDone!")
