"""
Module 01 — Functions & Decorators
Exercise 01 — First-Class Functions

WHAT YOU'RE BUILDING:
  Kernel launchers in Triton/CUTLASS are higher-order functions — they take
  compute functions and wrap them with grid setup, constant buffering, etc.
  This pattern is exactly how @cutlass.jit and @torch.compile work.

OBJECTIVE:
  - Understand functions as first-class objects (pass, return, store)
  - Build a simple "kernel launcher" wrapper pattern
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What happens when you pass a function as an argument to another function?
# Q2: Can you store functions in a dict and call them by key? What would the
#     syntax look like?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import Callable, Tuple

# Simulated kernel functions — in real work these would be Triton kernels
def matmul_fp32(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """Simulated FP32 matmul kernel launch."""
    return (M, N, K)

def matmul_fp16(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """Simulated FP16 matmul kernel launch."""
    return (M, N, K)

def matmul_int8(M: int, N: int, K: int) -> Tuple[int, int, int]:
    """Simulated INT8 matmul kernel launch."""
    return (M, N, K)

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Create a dict mapping dtype strings to kernel functions.
#              Keys: "fp32", "fp16", "int8"
#              Values: the corresponding matmul_* functions (not calls!)
# HINT: Don't add () after function names — you want the function object, not the result.

KERNEL_REGISTRY = {
    # TODO: fill in this dict
}

# TODO [EASY]: Write a launcher function that:
#              1. Takes dtype (str) and M, N, K (int) as args
#              2. Looks up the kernel from KERNEL_REGISTRY
#              3. Calls the kernel with M, N, K
#              4. Returns the result
# HINT: kernel_fn = KERNEL_REGISTRY[dtype], then call kernel_fn(M, N, K)

def launch_kernel(dtype: str, M: int, N: int, K: int) -> Tuple[int, int, int]:
    """Launch a matmul kernel by dtype."""
    # TODO: implement this
    pass

# TODO [MEDIUM]: Write a function that returns a *configured* kernel launcher.
#              This is the pattern behind functools.partial and currying.
#              Return a closure that has M, N, K baked in, only needs dtype.
# HINT: Inner function captures M, N, K from outer scope.

def make_fixed_size_launcher(M: int, N: int, K: int) -> Callable[[str], Tuple[int, int, int]]:
    """Return a launcher with M, N, K pre-configured."""
    # TODO: implement this closure pattern
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Did your predictions match? What happens when you call launch_kernel("fp16", 1024, 1024, 1024)?
# C2: How is this pattern used in real kernel frameworks? (Hint: think @triton.jit, dtype dispatch)

if __name__ == "__main__":
    # Test your implementation
    print("Testing launch_kernel...")
    result = launch_kernel("fp16", 1024, 1024, 1024)
    print(f"launch_kernel('fp16', 1024, 1024, 1024) = {result}")
    
    print("\nTesting make_fixed_size_launcher...")
    launcher = make_fixed_size_launcher(512, 512, 512)
    result = launcher("int8")
    print(f"launcher('int8') = {result}")
    
    print("\nAll tests passed!" if result == (512, 512, 512) else "Check your implementation!")
