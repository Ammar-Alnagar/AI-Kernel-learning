"""
Module 09 — Async Basics
Exercise 03 — Async Request Queue

WHAT YOU'RE BUILDING:
  Async queues coordinate concurrent tasks. In inference systems:
  - Rate-limiting API requests (e.g., 100 req/s limit)
  - Batching incoming inference requests
  - Producer-consumer patterns for async pipelines
  
  asyncio.Queue is the async equivalent of queue.Queue.

OBJECTIVE:
  - Use asyncio.Queue for task coordination
  - Implement rate limiting with semaphores
  - Build a simple async producer-consumer pattern
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the difference between asyncio.Queue and list for task coordination?
# Q2: How does asyncio.Semaphore limit concurrency?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import asyncio
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class InferenceRequest:
    """Simulated inference request."""
    request_id: str
    input_data: List[float]
    priority: int = 0

@dataclass
class InferenceResponse:
    """Simulated inference response."""
    request_id: str
    result: List[float]
    latency_ms: float

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Implement a rate-limited request processor.
#              Use asyncio.Semaphore to limit concurrent requests.
#              This is how you respect API rate limits.
# HINT: async with semaphore: process request

async def process_request(
    request: InferenceRequest,
    semaphore: asyncio.Semaphore
) -> InferenceResponse:
    """Process inference request with rate limiting."""
    # TODO: acquire semaphore, simulate processing, return response
    pass

# TODO [MEDIUM]: Implement producer that adds requests to queue.
#              Signal completion with None sentinel.
# HINT: await queue.put(request) for each, then queue.put(None)

async def producer(
    queue: asyncio.Queue,
    requests: List[InferenceRequest]
):
    """Add requests to queue, signal completion with None."""
    # TODO: put all requests, then put None sentinel
    pass

# TODO [MEDIUM]: Implement consumer that processes from queue.
#              Stop on None sentinel.
# HINT: while True: item = await queue.get(); if item is None: break

async def consumer(
    queue: asyncio.Queue,
    semaphore: asyncio.Semaphore,
    results: List[InferenceResponse]
):
    """Process requests from queue until None sentinel."""
    # TODO: get from queue, process, append to results
    pass

# TODO [HARD]: Wire together producer + consumers with asyncio.gather.
#              This is the full producer-consumer pattern.
# HINT: Run producer and multiple consumers with asyncio.gather

async def run_request_pipeline(
    requests: List[InferenceRequest],
    max_concurrent: int = 4
) -> List[InferenceResponse]:
    """Run full producer-consumer pipeline.
    
    Args:
        requests: Input requests to process
        max_concurrent: Maximum concurrent processing
    
    Returns:
        List of processed responses
    """
    queue = asyncio.Queue()
    semaphore = asyncio.Semaphore(max_concurrent)
    results: List[InferenceResponse] = []
    
    # TODO: run producer and consumer concurrently
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How does the semaphore limit concurrency?
# C2: Why use a queue instead of just gathering all tasks?

if __name__ == "__main__":
    print("Testing async request queue...\n")
    
    # Create test requests
    requests = [
        InferenceRequest(f"req-{i}", [float(i)] * 10)
        for i in range(10)
    ]
    
    async def main():
        print(f"Processing {len(requests)} requests (max 4 concurrent)...")
        
        responses = await run_request_pipeline(requests, max_concurrent=4)
        
        print(f"Processed {len(responses)} responses")
        for resp in responses[:3]:
            print(f"  {resp.request_id}: {resp.latency_ms:.1f}ms")
    
    asyncio.run(main())
    
    print("\nDone!")
