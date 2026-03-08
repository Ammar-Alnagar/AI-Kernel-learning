"""
Capstone — Kernel Benchmark Harness

A complete benchmark script tying together all Python concepts:
- CLI args via argparse (--kernel, --dtype, --sizes, --output)
- Config stored in @dataclass
- CUDA events for timing
- Dict comprehension for result collection
- CSV output via pathlib
- ncu invocation via subprocess
- Formatted results table

Usage:
    python benchmark_harness.py --kernel matmul --dtype float16 --sizes 1024 2048 4096 --output results.csv
"""

# ─────────────────────────────────────────────
# SETUP — Imports
# ─────────────────────────────────────────────
import argparse
import subprocess
import csv
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Define the benchmark config dataclass.
#              This stores all CLI arguments as a typed struct.
# HINT: @dataclass with fields for kernel, dtype, sizes, output, warmup, num_iters

@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    # TODO: add fields
    pass

# TODO [MEDIUM]: Create argument parser matching the config dataclass.
#              All CLI args should map to BenchmarkConfig fields.
# HINT: Same pattern as Module 07 ex01

def create_parser() -> argparse.ArgumentParser:
    """Create CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Kernel Benchmark Harness",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python benchmark_harness.py --kernel matmul --dtype float16 --sizes 1024 2048 4096
  python benchmark_harness.py --kernel attention --dtype float32 --sizes 512 1024 --output attn_results.csv
        """
    )
    # TODO: add arguments for kernel, dtype, sizes, output, warmup, num_iters
    pass

# TODO [EASY]: Parse args and convert to BenchmarkConfig.
# HINT: args = parser.parse_args(); return BenchmarkConfig(**vars(args))

def parse_config(argv: Optional[List[str]] = None) -> BenchmarkConfig:
    """Parse CLI args into BenchmarkConfig."""
    parser = create_parser()
    # TODO: implement
    pass

# TODO [MEDIUM]: Time a kernel using CUDA events.
#              This is the core timing primitive.
# HINT: Same pattern as Module 08 ex01

def time_kernel_cuda(
    kernel_fn,
    *args,
    warmup: int = 3,
    num_iters: int = 10
) -> tuple:
    """Time kernel with CUDA events.
    
    Returns:
        (mean_ms, std_ms)
    """
    if not torch.cuda.is_available():
        return (1.0, 0.1)
    
    # TODO: implement CUDA event timing
    pass

# TODO [MEDIUM]: Calculate TFLOPS for matmul.
#              FLOPS = 2 * M * N * K
# HINT: tflops = (2 * M * N * K) / (ms * 1e-3) / 1e12

def calculate_tflops(M: int, N: int, K: int, ms: float) -> float:
    """Calculate TFLOPS for matmul."""
    # TODO: implement
    pass

# TODO [MEDIUM]: Run benchmarks for all sizes.
#              Use list comprehension to collect results.
#              First size is the baseline for comparison.
# HINT: [BenchmarkResult(...) for size in sizes]

@dataclass
class BenchmarkResult:
    """Single benchmark result."""
    size: int
    ms: float
    tflops: float
    baseline_ratio: float = 1.0

def run_benchmarks(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run benchmarks for all sizes in config."""
    results = []
    baseline_ms = None
    
    # TODO: implement benchmark loop
    # For each size:
    #   1. Create tensors
    #   2. Time matmul
    #   3. Calculate TFLOPS
    #   4. Calculate baseline ratio
    #   5. Append result
    pass
    
    return results

# TODO [EASY]: Print results as formatted table.
#              Match the exact format from the spec.
# HINT: Use box-drawing characters and f-string field width

def print_results_table(results: List[BenchmarkResult]):
    """Print formatted results table.
    
    Format:
    ┌────────┬──────────┬───────────┬──────────────┐
    │  Size  │  Ms/iter │  TFLOPS   │  vs baseline │
    ├────────┼──────────┼───────────┼──────────────┤
    │  1024  │   0.21   │  102.4    │    1.0×      │
    └────────┴──────────┴───────────┴──────────────┘
    """
    # TODO: implement table printing
    pass

# TODO [EASY]: Save results to CSV using pathlib.
# HINT: Same pattern as Module 07 ex02

def save_results_csv(results: List[BenchmarkResult], output_path: str):
    """Save results to CSV file."""
    # TODO: implement
    pass

# TODO [MEDIUM]: Invoke ncu for the largest size.
#              This profiles the most interesting case.
# HINT: subprocess.run(["ncu", ...], capture_output=True)

def profile_largest_size(config: BenchmarkConfig):
    """Run ncu profile on largest size."""
    if not config.sizes:
        return
    
    largest = max(config.sizes)
    print(f"\nProfiling largest size ({largest}) with ncu...")
    
    # TODO: implement ncu invocation
    pass

# TODO [EASY]: Main entry point.
#              Tie everything together.
# HINT: config = parse_config(); results = run_benchmarks(config); print...; save...

def main(argv: Optional[List[str]] = None):
    """Main entry point."""
    # TODO: implement main flow
    pass

if __name__ == "__main__":
    main()
