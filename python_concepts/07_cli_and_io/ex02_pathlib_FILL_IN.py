"""
Module 07 — CLI and I/O
Exercise 02 — Pathlib

WHAT YOU'RE BUILDING:
  pathlib is the modern way to handle file paths. You'll use it for:
  - Saving benchmark results (CSV, JSON)
  - Loading model weights
  - Managing output directories
  It's safer and more readable than string concatenation.

OBJECTIVE:
  - Use Path for path manipulation
  - Create directories, check existence
  - Read/write text and CSV files
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between Path("a") / "b" and "a" + "/" + "b"?
# Q2: How do you check if a path exists with pathlib?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from pathlib import Path
from typing import List, Dict, Any
import csv

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Create output directory if it doesn't exist.
#              This is essential for benchmark scripts.
# HINT: path.mkdir(parents=True, exist_ok=True)

def ensure_output_dir(output_path: str) -> Path:
    """Ensure output directory exists, return Path object.
    
    Args:
        output_path: Path to output file (e.g., "results/benchmark.csv")
    
    Returns:
        Path object for the output file
    """
    path = Path(output_path)
    # TODO: create parent directory if needed
    pass

# TODO [EASY]: Write benchmark results to CSV.
#              Use csv.DictWriter for clean output.
# HINT: writer = csv.DictWriter(f, fieldnames=...); writer.writeheader(); writer.writerows(rows)

def write_results_csv(
    output_path: Path,
    results: List[Dict[str, Any]]
):
    """Write benchmark results to CSV.
    
    Args:
        output_path: Path to output CSV file
        results: List of dicts with keys: size, ms, tflops, baseline_ratio
    """
    # TODO: implement CSV writing
    pass

# TODO [MEDIUM]: Read results from CSV and return as list of dicts.
#              This is how you load previous benchmark results.
# HINT: reader = csv.DictReader(f); return list(reader)

def read_results_csv(output_path: Path) -> List[Dict[str, str]]:
    """Read benchmark results from CSV.
    
    Args:
        output_path: Path to input CSV file
    
    Returns:
        List of dicts (values are strings from CSV)
    """
    # TODO: implement CSV reading
    pass

# TODO [EASY]: Check if a file exists before writing.
#              Return a safe path (maybe with suffix added).
# HINT: path.exists()

def get_safe_output_path(base_path: str) -> Path:
    """Get a safe output path that doesn't overwrite existing files.
    
    If file exists, append _1, _2, etc. to filename.
    """
    path = Path(base_path)
    # TODO: implement collision avoidance
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why is pathlib better than string concatenation for paths?
# C2: How do you handle cross-platform path differences?

if __name__ == "__main__":
    print("Testing pathlib operations...\n")
    
    # Test ensure_output_dir
    output_path = ensure_output_dir("test_output/results.csv")
    print(f"Output path: {output_path}")
    print(f"Parent exists: {output_path.parent.exists()}\n")
    
    # Test write CSV
    test_results = [
        {"size": 1024, "ms": 0.21, "tflops": 102.4, "baseline_ratio": 1.0},
        {"size": 2048, "ms": 0.38, "tflops": 226.1, "baseline_ratio": 2.2},
    ]
    write_results_csv(output_path, test_results)
    print(f"Wrote {len(test_results)} results to {output_path}\n")
    
    # Test read CSV
    loaded = read_results_csv(output_path)
    print(f"Loaded results: {loaded}\n")
    
    print("Done!")
