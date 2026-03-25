# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 04: Tile Operations - Fill-in Code Exercise"""

import cuda.tile as ct
import torch

TILE_SIZE = 32
TILE_M = 16
TILE_N = 16


@ct.kernel
def affine_kernel(x, y, alpha: float, beta: float):
    """Compute y = x * alpha + beta."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN

    # FILL IN
    y_tile =  # FILL IN

    # FILL IN
    # FILL IN


@ct.kernel
def scalar_broadcast_add_kernel(x, y, scalar: float):
    """Add scalar to each tile element."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN

    # FILL IN
    y_tile =  # FILL IN

    # FILL IN
    # FILL IN


@ct.kernel
def transpose_2d_kernel(x2d, y2d):
    """Transpose each (TILE_M, TILE_N) tile."""
    # FILL IN
    bid_m =  # FILL IN
    bid_n =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN

    # FILL IN: transpose the tile
    y_tile =  # FILL IN

    # FILL IN: store to swapped output tile coordinates
    # FILL IN


@ct.kernel
def tile_center_kernel(x, y):
    """Center each tile by subtracting tile mean."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN

    # FILL IN: mean reduction over tile
    mean_val =  # FILL IN

    # FILL IN
    y_tile =  # FILL IN

    # FILL IN
    # FILL IN


def launch_affine(x: torch.Tensor, alpha: float, beta: float) -> torch.Tensor:
    y = torch.empty_like(x)
    grid = ((x.shape[0] + TILE_SIZE - 1) // TILE_SIZE, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, affine_kernel, (x, y, alpha, beta))
    return y


def launch_scalar_add(x: torch.Tensor, scalar: float) -> torch.Tensor:
    y = torch.empty_like(x)
    grid = ((x.shape[0] + TILE_SIZE - 1) // TILE_SIZE, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, scalar_broadcast_add_kernel, (x, y, scalar))
    return y


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 04.")
