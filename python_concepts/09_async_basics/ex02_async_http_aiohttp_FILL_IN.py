"""
Module 09 — Async Basics
Exercise 02 — Async HTTP with aiohttp

WHAT YOU'RE BUILDING:
  Async HTTP is essential for systems work:
  - Downloading model weights from S3/HuggingFace concurrently
  - Hitting multiple inference API endpoints in parallel
  - Async webhook notifications for job completion
  
  aiohttp is the standard async HTTP client.

OBJECTIVE:
  - Use aiohttp for async HTTP requests
  - Download multiple resources concurrently
  - Handle HTTP errors in async context
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: How does aiohttp differ from requests for concurrent downloads?
# Q2: Why do you need async with for aiohttp sessions?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import asyncio
# Note: Install with: pip install aiohttp
try:
    import aiohttp
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    print("aiohttp not installed. Install with: pip install aiohttp")

from typing import List, Dict, Optional

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [MEDIUM]: Download multiple URLs concurrently with aiohttp.
#              This is how you'd download sharded model weights.
#              Use async with aiohttp.ClientSession() as session:
# HINT: async with session.get(url) as response: return await response.text()

async def download_url(session: aiohttp.ClientSession, url: str) -> Optional[str]:
    """Download content from a single URL."""
    # TODO: implement with session.get()
    pass

async def download_multiple_urls(urls: List[str]) -> Dict[str, str]:
    """Download multiple URLs concurrently.
    
    Returns:
        Dict mapping URL -> content (or None if failed)
    """
    # TODO: use aiohttp.ClientSession and asyncio.gather
    pass

# TODO [EASY]: Download JSON from an API endpoint.
#              This is how you'd fetch model configs from a REST API.
# HINT: await response.json() instead of response.text()

async def fetch_json_api(session: aiohttp.ClientSession, url: str) -> Optional[dict]:
    """Fetch JSON from API endpoint."""
    # TODO: implement with response.json()
    pass

# TODO [MEDIUM]: Download with retry logic.
#              Transient failures are common in distributed systems.
# HINT: for attempt in range(max_retries): try/except with exponential backoff

async def download_with_retry(
    session: aiohttp.ClientSession,
    url: str,
    max_retries: int = 3,
    base_delay: float = 0.1
) -> Optional[str]:
    """Download URL with retry on failure."""
    # TODO: implement retry logic
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: How does aiohttp achieve concurrency vs threading?
# C2: When would you use requests (sync) vs aiohttp (async)?

if __name__ == "__main__":
    if not AIOHTTP_AVAILABLE:
        print("aiohttp not installed, skipping...")
    else:
        print("Testing async HTTP with aiohttp...\n")
        
        # Test URLs (using httpbin for testing)
        test_urls = [
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/1",
            "https://httpbin.org/delay/1",
        ]
        
        async def main():
            print(f"Downloading {len(test_urls)} URLs concurrently...")
            
            async with aiohttp.ClientSession() as session:
                results = await download_multiple_urls(test_urls)
                print(f"Downloaded {len([r for r in results if r])} URLs successfully")
        
        asyncio.run(main())
    
    print("\nDone!")
