"""
Module 11 — Logging and Debugging
Exercise 03 — Debugging with pdb and logging

WHAT YOU'RE BUILDING:
  Debugging kernel code requires tools:
  - pdb for interactive debugging
  - logging for post-mortem analysis
  - Assertions for invariant checking
  
  This is how you debug shape mismatches and CUDA errors.

OBJECTIVE:
  - Use pdb.set_trace() for interactive debugging
  - Add assertions for invariant checking
  - Configure logging for debug sessions
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What pdb commands do you use most? (n, s, c, p, l)
# Q2: When should you use assert vs raise explicit exception?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import logging
import torch
from typing import Tuple, Optional

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Add assertions to validate tensor shapes.
#              This catches bugs early with clear messages.
# HINT: assert condition, f"Expected ..., got ..."

def validate_matmul_inputs(
    a: torch.Tensor,
    b: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Validate inputs for matmul.
    
    Args:
        a: (M, K) tensor
        b: (K, N) tensor
    
    Returns:
        Validated tensors
    
    Raises:
        AssertionError: If shapes don't match expectations
    """
    # TODO: add assertions for:
    # - a and b are 2D tensors
    # - a.shape[1] == b.shape[0] (K dimension matches)
    # - tensors are on same device
    pass

# TODO [MEDIUM]: Add debug logging around key operations.
#              Log shapes, dtypes, devices at DEBUG level.
# HINT: logger.debug(f"a.shape={a.shape}, b.shape={b.shape}")

def debug_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matmul with debug logging.
    
    Logs input/output shapes, dtypes, devices.
    """
    # TODO: add debug logging before and after matmul
    pass

# TODO [EASY]: Add a breakpoint for interactive debugging.
#              Use pdb.set_trace() or breakpoint() (Python 3.7+)
# HINT: breakpoint() then use n, s, c, p commands

def matmul_with_breakpoint(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Matmul with debug breakpoint.
    
    Uncomment breakpoint() for interactive debugging.
    """
    # Validate
    a, b = validate_matmul_inputs(a, b)
    
    # TODO: add breakpoint() here for debugging
    # Uncomment when you want to debug interactively
    
    result = torch.matmul(a, b)
    
    logger.debug(f"Result shape: {result.shape}")
    return result

# TODO [MEDIUM]: Create a debug context manager that enables debug logging.
#              Temporarily set logger to DEBUG level.
# HINT: Use @contextmanager, save old level, set DEBUG, yield, restore

from contextlib import contextmanager

@contextmanager
def debug_context(logger: logging.Logger):
    """Context manager that enables debug logging temporarily."""
    # TODO: implement
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What's the difference between assert and explicit validation?
# C2: When do you use pdb vs logging for debugging?

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.DEBUG)
    
    print("Testing debugging tools...\n")
    
    # Test validation
    print("Testing shape validation...")
    a = torch.randn(10, 20)
    b = torch.randn(20, 30)
    
    try:
        validated = validate_matmul_inputs(a, b)
        print("  Validation passed ✓")
    except AssertionError as e:
        print(f"  Validation failed: {e}")
    
    # Test debug matmul
    print("\nTesting debug matmul...")
    result = debug_matmul(a, b)
    print(f"  Result shape: {result.shape}")
    
    # Test debug context
    print("\nTesting debug context...")
    with debug_context(logger):
        logger.debug("This is a debug message inside context")
    
    logger.debug("This should not appear (logger back to INFO)")
    
    print("\nDone!")
