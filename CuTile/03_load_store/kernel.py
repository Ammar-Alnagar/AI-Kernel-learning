# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 03: Load/Store - Fill-in Code Exercise
"""

import cuda.tile as ct
import torch

TILE_SIZE = 32
TILE_M = 16
TILE_N = 16


# =============================================================================
# EXERCISE 1: Basic Copy Kernel
# =============================================================================
@ct.kernel
def copy_kernel(src, dst):
    """Copy data tile-by-tile from src to dst."""
    # FILL IN: block ID along dim 0
    bid =  # FILL IN

    # FILL IN: load 1D tile from src
    tile =  # FILL IN

    # FILL IN: store tile into dst
    # FILL IN


# =============================================================================
# EXERCISE 2: Boundary-Safe Scaling
# =============================================================================
@ct.kernel
def padded_scale_kernel(src, dst, scale: float):
    """Scale src by `scale` with boundary-safe load behavior."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN: load with ZERO padding for out-of-bounds
    tile = ct.load(src, index=(bid,), shape=(TILE_SIZE,),
                   padding_mode=  # FILL IN
                   )

    # FILL IN
    out_tile =  # FILL IN

    # FILL IN
    # FILL IN


# =============================================================================
# EXERCISE 3: 2D Tile Copy
# =============================================================================
@ct.kernel
def copy_2d_kernel(src2d, dst2d):
    """Copy 2D matrix using a 2D grid and 2D tiles."""
    # FILL IN
    bid_m =  # FILL IN
    bid_n =  # FILL IN

    # FILL IN
    tile =  # FILL IN

    # FILL IN
    # FILL IN


# =============================================================================
# EXERCISE 4: Strided Downsample
# =============================================================================
@ct.kernel
def downsample_even_kernel(src, dst):
    """Store only even-indexed elements from src into dst."""
    bid = ct.bid(0)

    # FILL IN: load a contiguous tile from src
    tile =  # FILL IN

    # FILL IN: use reshape/slicing-like logic by writing every 2nd element
    # For this tutorial we keep logic simple with arithmetic on indices in host helper.
    # Here, compute output tile as-is and let host map expected check.
    out_tile =  # FILL IN

    # FILL IN
    # FILL IN


def launch_copy(src: torch.Tensor) -> torch.Tensor:
    """Host helper for Exercise 1."""
    out = torch.empty_like(src)
    grid = ((src.shape[0] + TILE_SIZE - 1) // TILE_SIZE, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, copy_kernel, (src, out))
    return out


def launch_padded_scale(src: torch.Tensor, scale: float) -> torch.Tensor:
    """Host helper for Exercise 2."""
    out = torch.empty_like(src)
    grid = ((src.shape[0] + TILE_SIZE - 1) // TILE_SIZE, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, padded_scale_kernel, (src, out, scale))
    return out


def launch_copy_2d(src2d: torch.Tensor) -> torch.Tensor:
    """Host helper for Exercise 3."""
    out = torch.empty_like(src2d)
    grid_m = (src2d.shape[0] + TILE_M - 1) // TILE_M
    grid_n = (src2d.shape[1] + TILE_N - 1) // TILE_N
    ct.launch(torch.cuda.current_stream(), (grid_m, grid_n, 1), copy_2d_kernel, (src2d, out))
    return out


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 03.")
