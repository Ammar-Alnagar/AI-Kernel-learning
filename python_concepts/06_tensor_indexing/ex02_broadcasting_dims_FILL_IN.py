"""
Module 06 — Tensor Indexing
Exercise 02 — Broadcasting & Dimensions

WHAT YOU'RE BUILDING:
  Broadcasting is how attention masks work, how you add bias to tensors,
  and how you scale Q/K/V matrices. Understanding dim manipulation
  is essential for transformer kernel implementation.

OBJECTIVE:
  - Master unsqueeze/squeeze for dim manipulation
  - Understand broadcasting rules
  - Practice with attention-style operations
"""

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What shape does (4, 1) + (1, 6) broadcast to?
# Q2: What's the difference between unsqueeze(0) and unsqueeze(-1)?

# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
import torch

# ─────────────────────────────────────────────
# FILL IN
# ─────────────────────────────────────────────

# TODO [EASY]: Add a batch dimension to a 2D tensor.
#              (M, N) -> (1, M, N)
# HINT: tensor.unsqueeze(0)

def add_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Add batch dimension."""
    # TODO: implement
    pass

# TODO [EASY]: Remove singleton dimensions.
#              (1, M, 1) -> (M,)
# HINT: tensor.squeeze()

def remove_singleton_dims(tensor: torch.Tensor) -> torch.Tensor:
    """Remove all singleton dimensions."""
    # TODO: implement
    pass

# TODO [MEDIUM]: Reshape for broadcasting: add dim at specific position.
#              (M, N) -> (1, M, 1, N) for attention-style broadcasting
# HINT: Use unsqueeze multiple times or view/reshape

def reshape_for_attention(tensor: torch.Tensor) -> torch.Tensor:
    """Reshape (M, N) -> (1, M, 1, N) for attention broadcasting."""
    # TODO: implement
    pass

# TODO [MEDIUM]: Apply scale and mask using broadcasting.
#              This is the core of attention score computation.
#              - Scale: scores / sqrt(dim)
#              - Mask: scores + mask * -1e9 (where mask is 0 or 1)
# HINT: Broadcasting handles the rest if shapes are compatible

def apply_attention_scale_and_mask(
    scores: torch.Tensor,
    mask: torch.Tensor,
    scale: float
) -> torch.Tensor:
    """Apply attention scaling and causal mask.
    
    Args:
        scores: (B, H, M, N) attention scores
        mask: (M, N) or (1, 1, M, N) binary mask (1 = mask out)
        scale: 1/sqrt(dim) scaling factor
    
    Returns:
        Scaled and masked scores
    """
    # TODO: implement
    # 1. Apply scale
    # 2. Apply mask: scores + mask * -1e9
    pass

# ─────────────────────────────────────────────
# CHECKPOINT — answer after running
# ─────────────────────────────────────────────
# C1: What shape does (B, H, M, N) + (M, N) broadcast to?
# C2: Why is -1e9 used for masking instead of 0?

if __name__ == "__main__":
    print("Testing broadcasting operations...\n")
    
    # Test batch dim
    t = torch.randn(3, 4)
    t_batched = add_batch_dim(t)
    print(f"Original: {t.shape} -> Batched: {t_batched.shape}\n")
    
    # Test squeeze
    t_singleton = torch.randn(1, 5, 1)
    t_squeezed = remove_singleton_dims(t_singleton)
    print(f"Singleton: {t_singleton.shape} -> Squeezed: {t_squeezed.shape}\n")
    
    # Test attention reshape
    t_attn = reshape_for_attention(t)
    print(f"Attention reshape: {t.shape} -> {t_attn.shape}\n")
    
    # Test attention mask
    scores = torch.randn(2, 4, 8, 8)  # (B, H, M, N)
    mask = torch.triu(torch.ones(8, 8), diagonal=1)  # Causal mask
    masked = apply_attention_scale_and_mask(scores, mask, scale=1.0 / 8.0**0.5)
    print(f"Masked scores shape: {masked.shape}")
    print(f"Masked values (should have -inf): {masked[0, 0, 0, :5]}\n")
    
    print("Done!")
