"""
Module 06 — Tensor Indexing
Exercise 03 — Attention Mask Indexing

WHAT YOU'RE BUILDING:
  Causal attention masks are upper triangular matrices. Creating and
  applying them efficiently is critical for transformer inference.
  This is exactly how FlashAttention and xformers handle masking.

OBJECTIVE:
  - Create causal masks with torch.triu
  - Apply masks to attention scores
  - Understand mask broadcasting across batch/heads
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What does torch.triu(torch.ones(4, 4), diagonal=1) produce?
# Q2: How does a (M, N) mask broadcast to (B, H, M, N)?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch
from typing import Tuple

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Create a causal mask for sequence length seq_len.
#              1s in upper triangle (positions to mask), 0s elsewhere.
# HINT: torch.triu(torch.ones(seq_len, seq_len), diagonal=1)

def create_causal_mask(seq_len: int) -> torch.Tensor:
    """Create causal attention mask.
    
    Returns:
        (seq_len, seq_len) mask with 1s in upper triangle
    """
    # TODO: implement
    pass

# TODO [MEDIUM]: Create a causal mask optimized for a specific device/dtype.
#              This is what you'd use in real kernel code.
# HINT: Create mask, then .to(device=device, dtype=dtype)

def create_causal_mask_device(
    seq_len: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16
) -> torch.Tensor:
    """Create causal mask on specific device/dtype."""
    # TODO: implement
    pass

# TODO [MEDIUM]: Apply causal mask to attention scores.
#              The mask value should be a large negative number.
# HINT: scores.masked_fill(mask.bool(), -1e9)

def apply_causal_mask(
    scores: torch.Tensor,
    mask: torch.Tensor
) -> torch.Tensor:
    """Apply causal mask to attention scores.
    
    Args:
        scores: (B, H, M, N) attention scores
        mask: (M, N) causal mask
    
    Returns:
        Masked scores with -1e9 in masked positions
    """
    # TODO: implement using masked_fill
    pass

# TODO [HARD]: Create a combined causal + padding mask.
#              Padding mask: 1 where tokens are valid, 0 where padded.
#              Combined: mask out both future positions AND padding.
# HINT: Combine masks with logical_or before applying

def create_combined_mask(
    seq_len: int,
    valid_lengths: torch.Tensor  # (B,) length of valid sequence for each batch
) -> torch.Tensor:
    """Create combined causal + padding mask.
    
    Args:
        seq_len: Maximum sequence length
        valid_lengths: (B,) valid sequence length for each batch item
    
    Returns:
        (B, 1, 1, seq_len) combined mask
    """
    # TODO: implement
    # 1. Create causal mask (seq_len, seq_len)
    # 2. Create padding mask from valid_lengths
    # 3. Combine with logical_or
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: Why use -1e9 instead of -inf for masking?
# C2: How does FlashAttention avoid materializing the full mask?

if __name__ == "__main__":
    print("Testing attention mask indexing...\n")
    
    # Test causal mask
    mask = create_causal_mask(4)
    print(f"Causal mask (4x4):\n{mask}\n")
    
    # Test device mask
    device_mask = create_causal_mask_device(4, torch.device("cpu"))
    print(f"Device mask dtype: {device_mask.dtype}\n")
    
    # Test apply mask
    scores = torch.randn(2, 4, 4, 4)  # (B, H, M, N)
    masked = apply_causal_mask(scores, mask)
    print(f"Masked scores shape: {masked.shape}")
    print(f"First row masked: {masked[0, 0, 0, :]}\n")
    
    print("Done!")
