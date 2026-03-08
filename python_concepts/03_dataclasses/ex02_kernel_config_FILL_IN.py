"""
Module 03 — Dataclasses
Exercise 02 — Kernel Config Struct

WHAT YOU'RE BUILDING:
  Real kernel configs have many fields: block sizes, stages, warps,
  matrix shapes, dtype, etc. Dataclasses keep them organized and typed.
  This is exactly how Triton's MetaParameters work.

OBJECTIVE:
  - Build a realistic kernel config dataclass
  - Use dataclasses with default values
  - Store configs in lists for benchmarking
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: How do you specify default values in a dataclass?
# Q2: What happens if you have fields with defaults before fields without defaults?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from dataclasses import dataclass, field
from typing import List, Tuple

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Define a MatmulConfig dataclass for GEMM kernel tuning.
#              Fields:
#              - M, N, K: int (matrix dimensions)
#              - block_m: int = 128
#              - block_n: int = 128
#              - block_k: int = 32
#              - num_stages: int = 3
#              - num_warps: int = 4
#              - dtype: str = "float16"
# HINT: Fields without defaults must come first!

@dataclass
class MatmulConfig:
    """GEMM kernel configuration."""
    # TODO: define fields (M, N, K first, then fields with defaults)
    pass

# TODO [EASY]: Generate a list of configs for benchmarking.
#              Vary block_m and block_n while keeping M, N, K fixed.
# HINT: [MatmulConfig(1024, 1024, 1024, block_m=bm, block_n=bn) for bm in [...] for bn in [...]]

def generate_benchmark_configs(M: int, N: int, K: int) -> List[MatmulConfig]:
    """Generate configs for benchmarking different block sizes."""
    block_sizes = [64, 128, 256]
    # TODO: list comprehension generating MatmulConfig instances
    pass

# TODO [MEDIUM]: Pretty-print configs as a table.
#              This is the pattern for benchmark result output.
# HINT: f-string with field width: f"{cfg.block_m:>8}"

def print_config_table(configs: List[MatmulConfig]):
    """Print configs as a formatted table."""
    # TODO: print header, then each config as a row
    # Format: | block_m | block_n | stages | warps | dtype |
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How many configs does generate_benchmark_configs(1024, 1024, 1024) produce?
# C2: Why put M, N, K before fields with defaults?

if __name__ == "__main__":
    print("Generating benchmark configs...")
    configs = generate_benchmark_configs(1024, 1024, 1024)
    print(f"Generated {len(configs)} configs\n")
    
    print("Config table:")
    print_config_table(configs)
    
    print("\nDone!")
