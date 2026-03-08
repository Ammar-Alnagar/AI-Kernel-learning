"""
Module 10 — Error Handling
Exercise 01 — Try/Except/Finally

WHAT YOU'RE BUILDING:
  Robust kernel code handles failures gracefully:
  - OOM errors during allocation → fallback to smaller batch
  - CUDA out of memory → clear cache, retry
  - Network failures during weight download → retry with backoff
  
  Understanding exception hierarchy is essential for systems code.

OBJECTIVE:
  - Use try/except/finally correctly
  - Catch specific exceptions, not bare except
  - Use finally for cleanup (like CUDA cache clear)
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between except Exception and except:?
# Q2: When does finally execute — even if there's a return in try?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from typing import Optional, List

CUDA_AVAILABLE = torch.cuda.is_available()

# Custom exception for kernel failures
class KernelExecutionError(Exception):
    """Raised when kernel execution fails."""
    pass

class OutOfMemoryError(Exception):
    """Raised when GPU/CPU memory allocation fails."""
    pass

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Handle a specific exception type.
#              Catch only the expected failure, not all exceptions.
# HINT: except ValueError: or except torch.cuda.OutOfMemoryError:

def safe_tensor_allocation(size: int) -> Optional[torch.Tensor]:
    """Try to allocate tensor, return None on failure."""
    # TODO: implement with try/except for specific exception
    pass

# TODO [MEDIUM]: Use finally for cleanup.
#              This is essential for GPU resource management.
# HINT: finally: torch.cuda.empty_cache() or cleanup code

def kernel_with_cleanup(data: torch.Tensor) -> torch.Tensor:
    """Run kernel, always cleanup even on error."""
    try:
        # Simulate kernel that might fail
        if data.numel() > 1_000_000_000:
            raise OutOfMemoryError("Simulated OOM")
        return data * 2
    # TODO: add except and finally blocks
    pass

# TODO [MEDIUM]: Implement retry with exponential backoff.
#              This is the standard pattern for transient failures.
# HINT: for attempt in range(max_retries): try/except with sleep

def operation_with_retry(
    operation_fn,
    max_retries: int = 3,
    base_delay: float = 0.1
):
    """Retry operation with exponential backoff.
    
    Args:
        operation_fn: Function to retry
        max_retries: Maximum retry attempts
        base_delay: Base delay between retries
    
    Returns:
        Result from operation_fn
    
    Raises:
        Last exception if all retries fail
    """
    import time
    last_exception = None
    
    # TODO: implement retry loop
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why catch specific exceptions instead of bare except:?
# C2: When would you use finally vs context manager for cleanup?

if __name__ == "__main__":
    print("Testing error handling...\n")
    
    # Test safe allocation
    print("Testing safe_tensor_allocation...")
    result = safe_tensor_allocation(100)
    print(f"Small allocation: {result.shape if result else 'Failed'}")
    
    # Test cleanup
    print("\nTesting kernel_with_cleanup...")
    try:
        data = torch.randn(100, 100)
        kernel_with_cleanup(data)
    except Exception as e:
        print(f"Caught: {e}")
    
    # Test retry
    print("\nTesting operation_with_retry...")
    attempt_count = 0
    
    def flaky_operation():
        nonlocal attempt_count
        attempt_count += 1
        if attempt_count < 3:
            raise KernelExecutionError(f"Transient failure #{attempt_count}")
        return "Success!"
    
    result = operation_with_retry(flaky_operation, max_retries=5)
    print(f"Result after {attempt_count} attempts: {result}")
    
    print("\nDone!")
