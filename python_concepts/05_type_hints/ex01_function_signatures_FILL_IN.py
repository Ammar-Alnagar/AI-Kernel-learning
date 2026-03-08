"""
Module 05 — Type Hints
Exercise 01 — Function Signatures

WHAT YOU'RE BUILDING:
  Type hints are essential for kernel APIs — they document tensor shapes,
  dtypes, and return types. IDEs use them for autocomplete on torch.Tensor.
  This is how torch library functions are annotated.

OBJECTIVE:
  - Annotate function parameters and return types
  - Use Optional, Union, List, Dict from typing
  - Understand how type hints improve IDE support
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: Do type hints affect runtime behavior in Python?
# Q2: What does Optional[int] mean vs just int?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import Optional, List, Dict, Tuple, Union
import torch

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Annotate this kernel config function.
#              - block_size: int
#              - num_stages: int (optional, default 3)
#              - Returns: Tuple[int, int]
# HINT: def fn(param: type, param2: type = default) -> ReturnType:

def get_kernel_config(block_size: int, num_stages: int = 3):
    """Get kernel config — needs type annotations."""
    # TODO: add type hints to signature
    return (block_size, num_stages)

# TODO [EASY]: Annotate this function that builds a config dict.
#              - sizes: List[int]
#              - Returns: Dict[str, int] mapping "M", "N", "K" to sizes
# HINT: List[int] and Dict[str, int] from typing

def build_shape_dict(sizes: List[int]) -> Dict[str, int]:
    """Build shape dict from sizes list."""
    # TODO: add type hints
    return {"M": sizes[0], "N": sizes[1], "K": sizes[2]}

# TODO [MEDIUM]: Annotate this function that handles multiple dtypes.
#              - dtype: Union[str, torch.dtype]
#              - Returns: str (canonical name)
# HINT: Union[A, B] means the parameter can be A or B

def canonicalize_dtype(dtype: Union[str, torch.dtype]) -> str:
    """Convert dtype to canonical string name."""
    # TODO: add type hints
    if isinstance(dtype, torch.dtype):
        return str(dtype).split(".")[-1]
    return dtype

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Do type hints change runtime behavior? What's their purpose?
# C2: Why are type hints especially useful for tensor APIs?

if __name__ == "__main__":
    print("Testing type-annotated functions...")
    
    config = get_kernel_config(128, 4)
    print(f"Config: {config}")
    
    shape = build_shape_dict([1024, 1024, 512])
    print(f"Shape: {shape}")
    
    dtype_str = canonicalize_dtype(torch.float16)
    print(f"Dtype: {dtype_str}")
    
    print("\nDone!")
