# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 06: Advanced Tiling - Fill-in Code Exercise"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

TILE_M = 16
TILE_N = 16
TILE_SIZE = 32


def xor_swizzle(tile_id: int, mask: int) -> int:
    """Simple host-side swizzle helper."""
    # FILL IN
    return  # FILL IN


@ct.kernel
def map_2d_kernel(x2d, y2d):
    """2D mapping kernel: y = x * 2."""
    # FILL IN
    bid_m =  # FILL IN
    bid_n =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN
    y_tile =  # FILL IN

    # FILL IN
    # FILL IN


@ct.kernel
def map_3d_batch_kernel(x3d, y3d, scale: float):
    """3D kernel for (batch, row, col) tile mapping."""
    # FILL IN
    bid_b =  # FILL IN
    bid_m =  # FILL IN
    bid_n =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN
    y_tile =  # FILL IN

    # FILL IN
    # FILL IN


@ct.kernel
def swizzled_1d_kernel(x, y, mask: ConstInt):
    """Use swizzled tile IDs for demonstration."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN
    swizzled_id =  # FILL IN

    # FILL IN
    tile =  # FILL IN

    # FILL IN
    # FILL IN


def launch_map_2d(x2d: torch.Tensor) -> torch.Tensor:
    y2d = torch.empty_like(x2d)
    grid = ((x2d.shape[0] + TILE_M - 1) // TILE_M,
            (x2d.shape[1] + TILE_N - 1) // TILE_N,
            1)
    ct.launch(torch.cuda.current_stream(), grid, map_2d_kernel, (x2d, y2d))
    return y2d


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 06.")
