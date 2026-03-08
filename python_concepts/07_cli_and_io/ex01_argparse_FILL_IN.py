"""
Module 07 — CLI and I/O
Exercise 01 — Argparse

WHAT YOU'RE BUILDING:
  Benchmark scripts need CLI args: --kernel, --dtype, --sizes, --output.
  argparse is the standard library for this. Every kernel engineer writes
  benchmark CLI tools — this is the foundation.

OBJECTIVE:
  - Parse CLI arguments with argparse
  - Handle required vs optional arguments
  - Convert args to config for benchmarking
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between positional and optional arguments?
# Q2: How does argparse handle --sizes 1024 2048 4096 (multiple values)?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import argparse
from typing import List

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Create an argument parser for a kernel benchmark script.
#              Arguments:
#              - --kernel: str, required, choices=["matmul", "attention", "layernorm"]
#              - --dtype: str, default="float16", choices=["float16", "float32", "int8"]
#              - --sizes: int, nargs="+", required (e.g., --sizes 1024 2048 4096)
#              - --output: str, default="results.csv"
#              - --warmup: int, default=3, help="warmup iterations"
#              - --num-iters: int, default=10, help="benchmark iterations"
# HINT: parser.add_argument("--name", type=..., default=..., choices=..., required=...)

def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for kernel benchmark."""
    parser = argparse.ArgumentParser(
        description="Kernel benchmark harness"
    )
    # TODO: add all arguments listed above
    pass

# TODO [EASY]: Parse arguments and return as a simple dict.
#              This is how you convert CLI args to a config object.
# HINT: args = parser.parse_args(); return vars(args)

def parse_benchmark_args(argv: List[str] = None) -> dict:
    """Parse CLI args and return as dict.
    
    Args:
        argv: Command line arguments (default: sys.argv[1:])
    
    Returns:
        Dict with kernel, dtype, sizes, output, warmup, num_iters
    """
    parser = create_parser()
    # TODO: parse and convert to dict
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How do you specify multiple sizes on the CLI?
# C2: What happens if you pass an invalid --dtype value?

if __name__ == "__main__":
    # Test with sample args
    test_args = [
        "--kernel", "matmul",
        "--dtype", "float16",
        "--sizes", "1024", "2048", "4096",
        "--output", "benchmark_results.csv",
        "--warmup", "5",
        "--num-iters", "20"
    ]
    
    print("Testing argument parsing...")
    config = parse_benchmark_args(test_args)
    
    print(f"Parsed config:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    print("\nDone!")
