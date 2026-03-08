"""
Module 08 — Timing and Benchmarking
Exercise 02 — Subprocess for NCU Profiling

WHAT YOU'RE BUILDING:
  NCU (NVIDIA Nsight Compute) is the gold standard for kernel profiling.
  You invoke it via subprocess to get detailed metrics: occupancy, memory
  bandwidth, instruction throughput. This is how you find bottlenecks.

OBJECTIVE:
  - Use subprocess to run external commands
  - Invoke ncu with appropriate flags
  - Parse or save ncu output
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between subprocess.run() and subprocess.Popen()?
# Q2: How do you capture stdout from a subprocess?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import subprocess
from pathlib import Path
from typing import Optional, List

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Run ncu on a Python script with a specific kernel.
#              This is how you profile custom kernels.
#              Key flags:
#              - --set full: Full metric set
#              - --target-processes all: Profile GPU
#              - -o output.nsys-rep: Output file
# HINT: subprocess.run(["ncu", "--set", "full", "python", "script.py"], capture_output=True)

def run_ncu_profile(
    script_path: str,
    output_path: Optional[str] = None,
    kernel_name: Optional[str] = None
) -> subprocess.CompletedProcess:
    """Run NCU profiler on a Python script.
    
    Args:
        script_path: Path to Python script to profile
        output_path: Optional path for .ncu-rep output
        kernel_name: Optional kernel name filter
    
    Returns:
        CompletedProcess with stdout, stderr, returncode
    """
    cmd = ["ncu", "--set", "full"]
    
    # TODO: add output path if specified
    # TODO: add kernel filter if specified
    # TODO: add script path
    # TODO: run and return result
    pass

# TODO [EASY]: Check if ncu is available in PATH.
#              This is how you gracefully handle missing tools.
# HINT: subprocess.run(["which", "ncu"], capture_output=True).returncode == 0

def is_ncu_available() -> bool:
    """Check if ncu is available in PATH."""
    # TODO: implement check
    pass

# TODO [MEDIUM]: Run a simple benchmark script with ncu.
#              Create a minimal profile command.
# HINT: Use run_ncu_profile with a test script

def profile_benchmark_script(
    script_path: str,
    output_dir: str = "profiles"
) -> dict:
    """Profile a benchmark script with ncu.
    
    Args:
        script_path: Path to benchmark script
        output_dir: Directory for profile output
    
    Returns:
        Dict with success status, output path, stdout
    """
    # TODO: ensure output dir exists
    # TODO: run ncu
    # TODO: return results dict
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What are the key metrics ncu provides that CUDA events don't?
# C2: Why might ncu profiling slow down kernel execution?

if __name__ == "__main__":
    print("Testing ncu subprocess...\n")
    
    # Check availability
    available = is_ncu_available()
    print(f"NCU available: {available}\n")
    
    if available:
        # Note: This would require an actual script to profile
        print("NCU is available. To profile a script:")
        print('  result = run_ncu_profile("my_benchmark.py")')
        print('  print(result.stdout.decode())')
    else:
        print("NCU not found. Install with NVIDIA Nsight Compute.")
    
    print("\nDone!")
