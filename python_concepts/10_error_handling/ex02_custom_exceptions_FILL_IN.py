"""
Module 10 — Error Handling
Exercise 02 — Custom Exceptions & Error Propagation

WHAT YOU'RE BUILDING:
  Custom exceptions document failure modes and enable selective handling:
  - KernelValidationError → input shape mismatch
  - KernelExecutionError → runtime failure
  - KernelConfigError → invalid configuration
  
  Exception hierarchies let you catch broad or specific failures.

OBJECTIVE:
  - Define custom exception classes
  - Use exception chaining (raise ... from ...)
  - Propagate errors with context preserved
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does `raise CustomError(...) from original_error` do?
# Q2: Why inherit from Exception instead of BaseException?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import Optional, Tuple
from dataclasses import dataclass

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Define a hierarchy of custom exceptions.
#              Base class → specific subclasses.
# HINT: class KernelError(Exception): pass
#       class KernelConfigError(KernelError): pass

class KernelError(Exception):
    """Base class for all kernel-related errors."""
    # TODO: implement base exception
    pass

class KernelConfigError(KernelError):
    """Raised when kernel configuration is invalid."""
    # TODO: implement
    pass

class KernelShapeError(KernelError):
    """Raised when tensor shapes don't match expectations."""
    # TODO: implement
    pass

class KernelExecutionError(KernelError):
    """Raised when kernel execution fails."""
    # TODO: implement
    pass

# TODO [MEDIUM]: Validate kernel config and raise custom exception.
#              Include helpful error message with actual vs expected.
# HINT: if invalid: raise KernelConfigError(f"Expected ..., got ...")

@dataclass
class KernelConfig:
    """Kernel configuration to validate."""
    block_size: int
    num_stages: int
    num_warps: int

def validate_kernel_config(config: KernelConfig):
    """Validate kernel configuration.
    
    Raises:
        KernelConfigError: If config is invalid
    """
    # TODO: implement validation with custom exceptions
    # Rules:
    # - block_size must be power of 2 (32, 64, 128, 256, 512, 1024)
    # - num_stages must be 2-6
    # - num_warps must be 4, 8, 16, or 32
    pass

# TODO [HARD]: Use exception chaining to preserve context.
#              This is critical for debugging in production.
# HINT: try: ... except OriginalError as e: raise NewError(...) from e

def execute_kernel(config: KernelConfig, input_data: list) -> list:
    """Execute kernel with proper error handling.
    
    Raises:
        KernelExecutionError: With original error chained
    """
    try:
        # Validate first
        validate_kernel_config(config)
        
        # Simulate execution that might fail
        if not input_data:
            raise ValueError("Empty input data")
        
        # Simulated success
        return [x * 2 for x in input_data]
        
    # TODO: catch and re-raise with chaining
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What information does exception chaining preserve?
# C2: When would you catch KernelError vs KernelConfigError specifically?

if __name__ == "__main__":
    print("Testing custom exceptions...\n")
    
    # Test valid config
    print("Testing valid config...")
    valid_config = KernelConfig(block_size=128, num_stages=3, num_warps=4)
    try:
        validate_kernel_config(valid_config)
        print("  Valid config passed ✓")
    except KernelConfigError as e:
        print(f"  Unexpected error: {e}")
    
    # Test invalid config
    print("\nTesting invalid config...")
    invalid_config = KernelConfig(block_size=100, num_stages=10, num_warps=5)
    try:
        validate_kernel_config(invalid_config)
    except KernelConfigError as e:
        print(f"  Caught expected error: {e}")
    
    # Test exception chaining
    print("\nTesting exception chaining...")
    try:
        execute_kernel(valid_config, [])
    except KernelExecutionError as e:
        print(f"  Caught: {e}")
        print(f"  Original cause: {e.__cause__}")
    
    print("\nDone!")
