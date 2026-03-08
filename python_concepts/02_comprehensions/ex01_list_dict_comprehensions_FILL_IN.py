"""
Module 02 — Comprehensions
Exercise 01 — List & Dict Comprehensions

WHAT YOU'RE BUILDING:
  Kernel benchmarking requires generating config grids (block sizes, shapes).
  Comprehensions build these grids concisely — same pattern as itertools.product
  but more readable for kernel engineers.

OBJECTIVE:
  - Build config grids with list comprehensions
  - Create lookup dicts with dict comprehensions
  - Filter configs with inline conditions
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does [x*2 for x in range(5)] produce?
# Q2: What does {k: v*2 for k, v in [("a", 1), ("b", 2)]} produce?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import List, Dict, Tuple

# Common block sizes in kernel tuning
BLOCK_SIZES = [32, 64, 128, 256, 512]
NUM_STAGES = [2, 3, 4, 5]

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Generate all (block_size, num_stages) pairs using a list comprehension.
#              This is your kernel config grid for tuning.
# HINT: [(bs, ns) for bs in BLOCK_SIZES for ns in NUM_STAGES]

def generate_config_grid() -> List[Tuple[int, int]]:
    """Generate all (block_size, num_stages) combinations."""
    # TODO: implement with nested list comprehension
    pass

# TODO [EASY]: Create a dict mapping block_size -> max_threads (block_size * 32).
#              This is useful for occupancy calculations.
# HINT: {bs: bs * 32 for bs in BLOCK_SIZES}

def build_block_size_lookup() -> Dict[int, int]:
    """Map block_size -> max_threads."""
    # TODO: implement with dict comprehension
    pass

# TODO [MEDIUM]: Filter configs to only those where block_size >= 128 AND num_stages <= 4.
#              This is how you prune the search space during tuning.
# HINT: [(bs, ns) for bs in BLOCK_SIZES for ns in NUM_STAGES if bs >= 128 and ns <= 4]

def generate_filtered_configs() -> List[Tuple[int, int]]:
    """Generate configs filtered by constraints."""
    # TODO: implement with conditional list comprehension
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How many configs are in the full grid vs the filtered grid?
# C2: How would you extend this to include num_warps in the grid?

if __name__ == "__main__":
    print("Config grid:")
    grid = generate_config_grid()
    print(f"Full grid: {len(grid)} configs")
    print(f"First 5: {grid[:5]}\n")
    
    print("Block size lookup:")
    lookup = build_block_size_lookup()
    print(f"{lookup}\n")
    
    print("Filtered configs:")
    filtered = generate_filtered_configs()
    print(f"Filtered: {len(filtered)} configs")
    print(f"{filtered}\n")
    
    print("Done!")
