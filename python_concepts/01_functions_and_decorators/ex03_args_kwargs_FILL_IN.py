"""
Module 01 — Functions & Decorators
Exercise 03 — *args and **kwargs

WHAT YOU'RE BUILDING:
  Kernel launch configs are flexible — block sizes, num_stages, num_warps, etc.
  **kwargs lets you pass arbitrary config without rigid signatures.
  This is how torch.cuda.launch() and triton.kernel.run() work.

OBJECTIVE:
  - Understand *args (positional) and **kwargs (keyword) unpacking
  - Build a flexible kernel launcher that accepts arbitrary config
  - See how real frameworks use this for extensible APIs
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between *args and **kwargs?
# Q2: If a function signature is def f(a, b, **kwargs), what happens when you
#     call f(1, 2, c=3, d=4)? What does kwargs contain?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import Dict, Any, Tuple

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Write a kernel launcher that accepts arbitrary launch config.
#              The function should:
#              1. Take M, N, K as required positional args
#              2. Accept **launch_config for block_size, num_stages, num_warps, etc.
#              3. Print the config being used
#              4. Return a tuple of (M, N, K, launch_config)
# HINT: def launch(M, N, K, **launch_config)

def launch_matmul(M: int, N: int, K: int, **launch_config) -> Tuple[int, int, int, Dict[str, Any]]:
    """Launch matmul with flexible config."""
    # TODO: implement — print config and return (M, N, K, launch_config)
    pass

# TODO [MEDIUM]: Write a function that forwards **kwargs to another function.
#              This is the pattern for wrapper functions that don't know
#              all the arguments ahead of time.
# HINT: def wrapper(M, N, K, **kwargs): return inner_fn(M, N, K, **kwargs)

def launch_with_defaults(M: int, N: int, K: int, **kwargs) -> Tuple[int, int, int, Dict[str, Any]]:
    """Launch matmul, filling in defaults for missing config."""
    # TODO: set defaults for block_size=128, num_stages=3, num_warps=4
    #       but allow kwargs to override them
    # HINT: config = {"block_size": 128, ...}; config.update(kwargs)
    pass

# TODO [EASY]: Understand dict unpacking with **.
#              What does this code do?
#              config = {"block_size": 256, "num_stages": 4}
#              launch_matmul(1024, 1024, 1024, **config)
# Q: What arguments does launch_matmul receive?

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What's in kwargs when you call launch_matmul(1024, 1024, 1024, block_size=256)?
# C2: Why is **kwargs essential for kernel APIs that evolve over time?

if __name__ == "__main__":
    print("Testing launch_matmul with **kwargs...")
    result = launch_matmul(1024, 1024, 1024, block_size=256, num_stages=4)
    print(f"Result: {result}\n")
    
    print("Testing launch_with_defaults...")
    result = launch_with_defaults(512, 512, 512, num_warps=8)
    print(f"Result: {result}\n")
    
    print("Testing dict unpacking...")
    config = {"block_size": 256, "num_stages": 4}
    result = launch_matmul(1024, 1024, 1024, **config)
    print(f"Result: {result}\n")
    
    print("Done!")
