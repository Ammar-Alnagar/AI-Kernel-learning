"""
Module 03 — Dataclasses
Exercise 01 — Dataclass as Struct

WHAT YOU'RE BUILDING:
  Kernel launch configs are plain data — no methods, no inheritance.
  @dataclass is Python's struct: typed, printable, comparable.
  This is how Triton/CUTLASS store kernel metadata.

OBJECTIVE:
  - Use @dataclass as a plain struct (no methods)
  - Understand fields with type hints
  - See how dataclasses replace dict configs
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does @dataclass add to a class automatically?
# Q2: Why is a dataclass better than a dict for kernel configs?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from dataclasses import dataclass
from typing import Optional

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Define a KernelConfig dataclass with fields:
#              - block_size: int
#              - num_stages: int
#              - num_warps: int
#              - name: Optional[str] = None  (default None)
# HINT: @dataclass on top, then field: type = default

@dataclass
class KernelConfig:
    """Kernel launch configuration struct."""
    # TODO: define fields here
    pass

# TODO [EASY]: Create instances and compare them.
#              Dataclasses have auto-generated __eq__ and __repr__.
# HINT: cfg1 = KernelConfig(128, 3, 4); cfg2 = KernelConfig(128, 3, 4)

def test_dataclass_features():
    """Test auto-generated methods."""
    # TODO: create two identical configs and compare them
    cfg1 = None  # TODO: replace
    cfg2 = None  # TODO: replace
    
    print(f"cfg1: {cfg1}")
    print(f"cfg2: {cfg2}")
    print(f"cfg1 == cfg2: {cfg1 == cfg2}")
    
    return cfg1, cfg2

# TODO [MEDIUM]: Convert a dict to a dataclass using **.
#              This is how you migrate from dict configs to typed configs.
# HINT: config_dict = {"block_size": 128, ...}; KernelConfig(**config_dict)

def dict_to_dataclass(config_dict: dict) -> KernelConfig:
    """Convert dict config to dataclass."""
    # TODO: unpack dict into KernelConfig constructor
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What methods does @dataclass auto-generate?
# C2: Why is type safety important for kernel configs?

if __name__ == "__main__":
    print("Testing dataclass features...")
    cfg1, cfg2 = test_dataclass_features()
    
    print("\nConverting dict to dataclass...")
    config_dict = {"block_size": 256, "num_stages": 4, "num_warps": 8}
    cfg = dict_to_dataclass(config_dict)
    print(f"Converted: {cfg}\n")
    
    print("Done!")
