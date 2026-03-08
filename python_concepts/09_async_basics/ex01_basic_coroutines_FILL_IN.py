"""
Module 09 — Async Basics
Exercise 01 — Basic Coroutines

WHAT YOU'RE BUILDING:
  Async/await enables concurrent I/O operations. In systems work:
  - Fetching multiple model configs from remote storage concurrently
  - Hitting multiple inference endpoints in parallel
  - Async metrics collection during benchmarking
  
  This is how async inference servers (vLLM, TGI) handle concurrent requests.

OBJECTIVE:
  - Understand async/await syntax
  - Run coroutines concurrently with asyncio.gather()
  - Know when async helps (I/O-bound) vs hurts (CPU-bound)
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between a regular function and a coroutine (async def)?
# Q2: Why doesn't calling an async function directly execute it?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import asyncio
import time
from typing import List, Dict

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Write a simple coroutine that simulates fetching model metadata.
#              Use asyncio.sleep() to simulate network delay.
# HINT: async def fetch_model_info(model_id: str) -> dict:
#       await asyncio.sleep(0.1)
#       return {"id": model_id, ...}

async def fetch_model_info(model_id: str) -> Dict[str, str]:
    """Simulate fetching model metadata from remote storage."""
    # TODO: implement with asyncio.sleep
    pass

# TODO [MEDIUM]: Run multiple fetches concurrently with asyncio.gather().
#              This is the key pattern for concurrent I/O.
# HINT: results = await asyncio.gather(fetch("a"), fetch("b"), fetch("c"))

async def fetch_multiple_models(model_ids: List[str]) -> List[Dict[str, str]]:
    """Fetch multiple model infos concurrently."""
    # TODO: implement with asyncio.gather
    pass

# TODO [EASY]: Measure the speedup from concurrent vs sequential fetches.
#              This demonstrates when async is useful.
# HINT: time.time() before/after, compare sequential vs concurrent

async def benchmark_fetch():
    """Compare sequential vs concurrent fetching."""
    model_ids = ["model1", "model2", "model3", "model4", "model5"]
    
    # Sequential
    start = time.time()
    # TODO: fetch one by one with await
    sequential_time = time.time() - start
    
    # Concurrent
    start = time.time()
    # TODO: fetch all concurrently with gather
    concurrent_time = time.time() - start
    
    print(f"Sequential: {sequential_time:.2f}s")
    print(f"Concurrent: {concurrent_time:.2f}s")
    print(f"Speedup: {sequential_time / concurrent_time:.1f}x")

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What speedup did you observe? Why isn't it 5x?
# C2: Would async help for CPU-bound kernel computation? Why or why not?

if __name__ == "__main__":
    print("Testing async coroutines...\n")
    
    # Run the benchmark
    asyncio.run(benchmark_fetch())
    
    print("\nDone!")
