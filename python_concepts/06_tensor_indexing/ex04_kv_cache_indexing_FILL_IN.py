"""
Module 06 — Tensor Indexing
Exercise 04 — KV Cache Indexing

WHAT YOU'RE BUILDING:
  KV caching is essential for efficient LLM inference. You store
  K/V tensors and update them token-by-token. Efficient indexing
  into the cache is critical for inference kernels.

OBJECTIVE:
  - Understand KV cache tensor layout
  - Practice token-by-token cache updates
  - Implement cache slicing for attention
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What's the typical shape of a KV cache tensor? (B, seq, H, D) or (B, H, seq, D)?
# Q2: How do you update position 5 in a sequence cache?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from typing import Tuple, Optional
from dataclasses import dataclass

@dataclass
class KVCacheConfig:
    """KV cache configuration."""
    batch_size: int
    num_heads: int
    head_dim: int
    max_seq_len: int

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Initialize KV cache tensors.
#              Shape: (batch_size, num_heads, max_seq_len, head_dim)
# HINT: torch.zeros(config.batch_size, config.num_heads, config.max_seq_len, config.head_dim)

def init_kv_cache(config: KVCacheConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    """Initialize K and V cache tensors.
    
    Returns:
        (key_cache, value_cache) each of shape (B, H, max_seq, head_dim)
    """
    # TODO: implement
    pass

# TODO [EASY]: Update cache at a specific position.
#              This is done token-by-token during generation.
# HINT: cache[:, :, position:position+1, :] = new_kv

def update_cache_position(
    cache: torch.Tensor,
    new_kv: torch.Tensor,
    position: int
) -> torch.Tensor:
    """Update cache at specific position.
    
    Args:
        cache: (B, H, max_seq, head_dim) cache
        new_kv: (B, H, 1, head_dim) new key or value
        position: position to update
    
    Returns:
        Updated cache (in-place modification, returned for convenience)
    """
    # TODO: implement
    pass

# TODO [MEDIUM]: Slice cache for attention computation.
#              Get all tokens up to current position.
# HINT: cache[:, :, :current_pos+1, :]

def slice_cache_for_attention(
    cache: torch.Tensor,
    current_pos: int
) -> torch.Tensor:
    """Slice cache up to current position for attention.
    
    Args:
        cache: (B, H, max_seq, head_dim) cache
        current_pos: current generation position
    
    Returns:
        (B, H, current_pos+1, head_dim) sliced cache
    """
    # TODO: implement
    pass

# TODO [HARD]: Implement a full KV cache update + attention prep.
#              This is the pattern used in LLM inference loops.
# HINT: Combine update and slice operations

def kv_cache_forward(
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    new_key: torch.Tensor,
    new_value: torch.Tensor,
    position: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Update KV cache and return sliced tensors for attention.
    
    Args:
        key_cache: (B, H, max_seq, head_dim)
        value_cache: (B, H, max_seq, head_dim)
        new_key: (B, H, 1, head_dim)
        new_value: (B, H, 1, head_dim)
        position: current position
    
    Returns:
        (keys_for_attn, values_for_attn) each (B, H, position+1, head_dim)
    """
    # TODO: implement
    # 1. Update key cache
    # 2. Update value cache
    # 3. Slice both for attention
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why is KV cache layout (B, H, seq, D) better than (B, seq, H, D)?
# C2: How does PagedAttention (vLLM) change this indexing pattern?

if __name__ == "__main__":
    print("Testing KV cache indexing...\n")
    
    # Setup
    config = KVCacheConfig(
        batch_size=2,
        num_heads=4,
        head_dim=64,
        max_seq_len=512
    )
    
    # Init cache
    key_cache, value_cache = init_kv_cache(config)
    print(f"Key cache shape: {key_cache.shape}")
    print(f"Value cache shape: {value_cache.shape}\n")
    
    # Simulate token generation
    for pos in range(3):
        new_k = torch.randn(config.batch_size, config.num_heads, 1, config.head_dim)
        new_v = torch.randn(config.batch_size, config.num_heads, 1, config.head_dim)
        
        keys, values = kv_cache_forward(key_cache, value_cache, new_k, new_v, pos)
        print(f"Position {pos}: attention input shape = {keys.shape}")
    
    print("\nDone!")
