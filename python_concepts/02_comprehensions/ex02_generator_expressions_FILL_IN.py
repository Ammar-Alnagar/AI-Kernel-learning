"""
Module 02 — Comprehensions
Exercise 02 — Generator Expressions

WHAT YOU'RE BUILDING:
  When benchmarking 1000s of kernel configs, you don't want to materialize
  all configs in memory. Generator expressions (like range()) yield lazily.
  This is critical for large-scale kernel sweeps.

OBJECTIVE:
  - Understand generator expressions vs list comprehensions
  - Build memory-efficient config iterators
  - Use generators with sum(), max(), min()
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between [x for x in range(1000000)] and (x for x in range(1000000))?
# Q2: Why would you use a generator expression with sum() instead of a list comprehension?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
from typing import Generator, Tuple

BLOCK_SIZES = [32, 64, 128, 256, 512, 1024, 2048]
NUM_STAGES = [2, 3, 4, 5, 6]
NUM_WARPS = [4, 8, 16]

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Write a generator expression that yields (block_size, num_stages, num_warps) tuples.
#              Don't materialize the full grid — yield lazily.
# HINT: Use parentheses () not brackets [] for generator expression

def config_generator() -> Generator[Tuple[int, int, int], None, None]:
    """Yield all (block_size, num_stages, num_warps) configs lazily."""
    # TODO: implement with generator expression
    # HINT: return ((bs, ns, nw) for bs in BLOCK_SIZES for ns in NUM_STAGES for nw in NUM_WARPS)
    pass

# TODO [EASY]: Use a generator expression with sum() to count total configs.
#              This avoids materializing the full list.
# HINT: sum(1 for _ in config_generator())

def count_configs() -> int:
    """Count total configs without materializing the list."""
    # TODO: implement with generator expression and sum()
    pass

# TODO [MEDIUM]: Find the config with maximum "score" (block_size * num_stages * num_warps)
#              using a generator expression with max().
#              This is the pattern for finding best config during tuning.
# HINT: max(config_generator(), key=lambda c: c[0] * c[1] * c[2])

def find_max_score_config() -> Tuple[int, int, int]:
    """Find config with maximum score using generator."""
    # TODO: implement with max() and generator expression
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How many configs are there? What's the max-score config?
# C2: Why is generator expression better than list comprehension for 10000s of configs?

if __name__ == "__main__":
    print("Counting configs lazily...")
    count = count_configs()
    print(f"Total configs: {count}\n")
    
    print("Finding max-score config...")
    best = find_max_score_config()
    print(f"Best config: {best}")
    print(f"Score: {best[0] * best[1] * best[2]}\n")
    
    print("Memory check: generator doesn't materialize all configs")
    gen = config_generator()
    print(f"Generator object: {gen}")
    print(f"First 3 configs: {[next(gen) for _ in range(3)]}\n")
    
    print("Done!")
