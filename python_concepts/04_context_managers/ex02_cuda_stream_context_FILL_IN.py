"""
Module 04 — Context Managers
Exercise 02 — CUDA Stream Context

WHAT YOU'RE BUILDING:
  CUDA streams are managed with context managers in PyTorch.
  `with torch.cuda.stream(s):` ensures kernels launch on the right stream.
  This pattern is everywhere: device selection, autocast, profiler regions.

OBJECTIVE:
  - Use torch.cuda.stream as a context manager
  - Understand device context with torch.cuda.device
  - Build a custom context manager for kernel launch config
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does `with torch.cuda.device(0):` do?
# Q2: If you create a tensor inside `with torch.cuda.stream(s):`, what stream is it associated with?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from typing import Optional
from contextlib import contextmanager

# Check CUDA availability
CUDA_AVAILABLE = torch.cuda.is_available()

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Create a stream and launch kernels on it.
#              Use torch.cuda.stream() as a context manager.
# HINT: with torch.cuda.stream(stream): tensor = torch.ones(...)

def test_stream_context():
    """Test CUDA stream context manager."""
    if not CUDA_AVAILABLE:
        print("CUDA not available, skipping...")
        return None
    
    # TODO: Create a stream, use it in a with block, create a tensor
    pass

# TODO [MEDIUM]: Write a context manager using @contextlib.contextmanager decorator.
#              This is an alternative to the class-based approach.
#              Yield control, then cleanup on exit.
# HINT: @contextmanager decorator, use yield, cleanup after yield

@contextmanager
def kernel_launch_context(kernel_name: str):
    """Context manager that prints kernel launch/complete messages."""
    # TODO: print "Launching {kernel_name}...", yield, then print "Complete"
    pass

# TODO [EASY]: Use the kernel_launch_context.
#              What gets printed?

def test_kernel_context():
    """Test the kernel launch context manager."""
    with kernel_launch_context("matmul_fp16"):
        # Simulated kernel work
        pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What's the difference between class-based and @contextmanager context managers?
# C2: Why is context manager pattern essential for CUDA resource management?

if __name__ == "__main__":
    print("Testing stream context...")
    test_stream_context()
    
    print("\nTesting kernel launch context...")
    test_kernel_context()
    
    print("\nDone!")
