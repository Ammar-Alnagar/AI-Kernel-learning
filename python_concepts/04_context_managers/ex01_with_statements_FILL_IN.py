"""
Module 04 — Context Managers
Exercise 01 — With Statements

WHAT YOU'RE BUILDING:
  Context managers handle setup/teardown automatically. In kernel work:
  CUDA device selection, stream creation, profiler regions, etc.
  The `with` statement ensures cleanup even on errors.

OBJECTIVE:
  - Understand the context manager protocol (__enter__, __exit__)
  - Use existing context managers (torch.cuda.device, profiler)
  - See how real frameworks use this for resource management
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does `with open("file.txt") as f:` do automatically?
# Q2: What happens to cleanup code if an exception is raised inside a with block?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import time
from typing import Optional, Any

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Implement a simple context manager class for timing.
#              This mimics torch.profiler.record_function() pattern.
#              __enter__: record start time, return self
#              __exit__: record end time, print elapsed
# HINT: __exit__(self, exc_type, exc_val, exc_tb) — return False to propagate exceptions

class Timer:
    """Context manager for timing code blocks."""
    
    def __init__(self, name: str = "block"):
        self.name = name
        self.elapsed: Optional[float] = None
    
    def __enter__(self):
        # TODO: record start time
        pass
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # TODO: record end time, print elapsed
        # Return False to not suppress exceptions
        pass

# TODO [EASY]: Use the Timer context manager.
#              What gets printed when this runs?

def test_timer():
    """Test the Timer context manager."""
    with Timer("test_block"):
        time.sleep(0.1)  # Simulate work
    
    # TODO: What gets printed? What is the elapsed time approximately?

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What protocol does a context manager implement?
# C2: Why is `with` safer than try/finally for cleanup?

if __name__ == "__main__":
    print("Testing Timer context manager...")
    test_timer()
    
    print("\nTesting exception handling...")
    try:
        with Timer("error_block"):
            time.sleep(0.05)
            raise ValueError("Test error")
    except ValueError:
        print("Exception propagated correctly")
    
    print("\nDone!")
