"""
Module 01 — Functions & Decorators
Exercise 02 — Decorators

WHAT YOU'RE BUILDING:
  Decorators are how @torch.compile, @triton.jit, and @cutlass.jit work.
  They wrap a kernel function with setup/teardown, caching, or compilation.
  You'll build a simple timing decorator — same pattern, simpler code.

OBJECTIVE:
  - Understand @decorator syntax as function wrapping
  - Build a decorator that adds timing around kernel execution
  - See how real frameworks use this for compilation caching
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does @my_decorator do to a function? Is it syntax sugar for something?
# Q2: Why does the wrapper function need *args and **kwargs?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import time
from typing import Callable, Any

def simulated_kernel(x: int, y: int) -> int:
    """Simulated kernel that does some work."""
    time.sleep(0.01)  # Simulate compute
    return x + y

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Write a timing decorator.
#              It should:
#              1. Accept a function fn
#              2. Return a wrapper function
#              3. Wrapper times fn execution and prints elapsed time
#              4. Wrapper returns fn's result
# HINT: Use time.perf_counter() before and after fn(*args, **kwargs)

def timed(fn: Callable) -> Callable:
    """Decorator that times function execution."""
    # TODO: implement the wrapper pattern
    pass

# Apply the decorator
@timed
def matmul_simulated(M: int, N: int, K: int) -> int:
    """Simulated matmul — decorated with @timed."""
    time.sleep(0.001 * (M // 128))  # Simulate compute scaling with M
    return M * N * K

# TODO [MEDIUM]: Write a decorator factory — a decorator that takes arguments.
#              This is how @torch.compile(fullgraph=True) works.
#              Return a decorator that conditionally applies timing.
# HINT: decorator_factory(enabled) returns a decorator, which returns a wrapper.

def conditional_timing(enabled: bool) -> Callable:
    """Decorator factory: @conditional_timing(enabled=True) adds timing."""
    # TODO: implement the nested pattern
    def decorator(fn: Callable) -> Callable:
        # TODO: implement inner decorator
        pass
    return decorator

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What does @timed expand to without the @ syntax? (Hint: matmul_simulated = ...)
# C2: How would you use this pattern to add @triton.jit compilation caching?

if __name__ == "__main__":
    print("Testing @timed decorator...")
    result = matmul_simulated(1024, 1024, 1024)
    print(f"Result: {result}\n")
    
    print("Testing @conditional_timing...")
    
    @conditional_timing(enabled=True)
    def kernel_with_timing(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    @conditional_timing(enabled=False)
    def kernel_without_timing(x: int) -> int:
        time.sleep(0.01)
        return x * 2
    
    print("With timing:")
    kernel_with_timing(100)
    
    print("Without timing (should be silent):")
    kernel_without_timing(100)
    
    print("\nDone!")
