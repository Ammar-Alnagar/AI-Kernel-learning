"""
Module 12 — Testing
Exercise 03 — Testing Benchmark Harness

WHAT YOU'RE BUILDING:
  Test your benchmark harness itself:
  - Mock CUDA events for testing without GPU
  - Test CSV output format
  - Test CLI argument parsing
  
  This ensures your testing infrastructure works.

OBJECTIVE:
  - Use unittest.mock for mocking
  - Test file I/O without side effects
  - Test CLI parsing with different inputs
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between mock.MagicMock and mock.Mock?
# Q2: How do you test file output without creating real files?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import tempfile
from pathlib import Path
from typing import List, Dict

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Test CLI argument parsing.
#              Verify all arguments are parsed correctly.
# HINT: from ex01_argparse_FILL_IN import parse_benchmark_args

def test_parse_benchmark_args():
    """Test CLI argument parsing."""
    # Import from Module 07
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "07_cli_and_io"))
    
    try:
        from ex01_argparse_FILL_IN import parse_benchmark_args
    except ImportError:
        print("Module 07 not found, skipping...")
        return
    
    test_args = [
        "--kernel", "matmul",
        "--dtype", "float16",
        "--sizes", "1024", "2048",
        "--output", "test.csv"
    ]
    
    # TODO: parse args and verify values
    pass

# TODO [MEDIUM]: Test CSV output with temporary file.
#              Use tempfile to avoid side effects.
# HINT: with tempfile.TemporaryDirectory() as tmpdir: ...

def test_write_results_csv():
    """Test CSV output writing."""
    # Import from Module 07
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "07_cli_and_io"))
    
    try:
        from ex02_pathlib_FILL_IN import write_results_csv, read_results_csv
    except ImportError:
        print("Module 07 not found, skipping...")
        return
    
    test_results = [
        {"size": 1024, "ms": 0.21, "tflops": 102.4},
        {"size": 2048, "ms": 0.38, "tflops": 226.1},
    ]
    
    # TODO: write to temp file, read back, verify
    pass

# TODO [HARD]: Mock CUDA events for testing without GPU.
#              Use unittest.mock to simulate torch.cuda.Event.
# HINT: from unittest.mock import MagicMock, patch
#       with patch('torch.cuda.Event') as mock_event: ...

def test_benchmark_without_gpu():
    """Test benchmark harness without actual GPU."""
    from unittest.mock import MagicMock, patch
    
    # TODO: mock torch.cuda.Event, torch.cuda.synchronize
    # Verify benchmark code calls the right methods
    pass

# TODO [EASY]: Test results table formatting.
#              Verify table output matches expected format.
# HINT: Capture stdout with capsys fixture or io.StringIO

def test_results_table_format():
    """Test results table formatting."""
    # Import from Module 08
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "08_timing_and_benchmarking"))
    
    try:
        from ex03_benchmark_harness_FILL_IN import BenchmarkResult, print_results_table
    except ImportError:
        print("Module 08 not found, skipping...")
        return
    
    test_results = [
        BenchmarkResult(size=1024, ms=0.21, tflops=102.4, baseline_ratio=1.0),
        BenchmarkResult(size=2048, ms=0.38, tflops=226.1, baseline_ratio=2.2),
    ]
    
    # TODO: capture output and verify format
    # Should contain box-drawing characters and aligned columns
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why use tempfile for testing file I/O?
# C2: What's the benefit of mocking CUDA for testing?

if __name__ == "__main__":
    print("Testing benchmark harness...\n")
    
    print("Testing CLI parsing...")
    test_parse_benchmark_args()
    
    print("\nTesting CSV output...")
    test_write_results_csv()
    
    print("\nTesting without GPU...")
    test_benchmark_without_gpu()
    
    print("\nTesting table format...")
    test_results_table_format()
    
    print("\nDone!")
