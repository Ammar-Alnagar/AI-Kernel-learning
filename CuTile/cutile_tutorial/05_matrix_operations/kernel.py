# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 05: Matrix Operations - Fill-in Code Exercise"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

TILE_M = 32
TILE_N = 32
TILE_K = 32


@ct.kernel
def single_tile_matmul_kernel(a, b, c,
                              tile_m: ConstInt,
                              tile_n: ConstInt,
                              tile_k: ConstInt):
    """Compute one output tile using ct.matmul()."""
    # FILL IN
    bid_m =  # FILL IN
    bid_n =  # FILL IN

    # FILL IN
    a_tile =  # FILL IN
    b_tile =  # FILL IN

    # FILL IN
    c_tile =  # FILL IN

    # FILL IN
    # FILL IN


@ct.kernel
def tiled_matmul_kloop_kernel(a, b, c,
                              tile_m: ConstInt,
                              tile_n: ConstInt,
                              tile_k: ConstInt,
                              k_tiles: ConstInt):
    """Accumulate matmul across multiple K tiles."""
    # FILL IN
    bid_m =  # FILL IN
    bid_n =  # FILL IN

    # FILL IN: initialize accumulator tile to zeros
    acc =  # FILL IN

    # FILL IN: loop over K tiles and accumulate
    # for k_id in range(k_tiles):
    #   load a/b tiles
    #   acc = acc + ct.matmul(a_tile, b_tile)

    # FILL IN: store acc tile
    # FILL IN


@ct.kernel
def mma_style_kernel(a, b, c,
                     tile_m: ConstInt,
                     tile_n: ConstInt,
                     tile_k: ConstInt,
                     k_tiles: ConstInt):
    """Tensor Core style accumulation using ct.mma()."""
    # FILL IN
    bid_m =  # FILL IN
    bid_n =  # FILL IN

    # FILL IN
    acc =  # FILL IN

    # FILL IN
    # for k_id in range(k_tiles):
    #   a_frag = ...
    #   b_frag = ...
    #   acc = ct.mma(a_frag, b_frag, acc)

    # FILL IN
    # FILL IN


def launch_tiled_matmul(a: torch.Tensor, b: torch.Tensor,
                        tile_m: int = TILE_M,
                        tile_n: int = TILE_N,
                        tile_k: int = TILE_K) -> torch.Tensor:
    """Host helper for tiled matmul kernel."""
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match")

    m, k = a.shape
    _, n = b.shape

    c = torch.empty((m, n), dtype=a.dtype, device=a.device)

    grid_m = (m + tile_m - 1) // tile_m
    grid_n = (n + tile_n - 1) // tile_n
    k_tiles = (k + tile_k - 1) // tile_k

    grid = (grid_m, grid_n, 1)

    # FILL IN: launch tiled_matmul_kloop_kernel
    # FILL IN

    return c


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 05.")
