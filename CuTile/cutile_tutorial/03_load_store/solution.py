# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""
Module 03: Load/Store - Solution
"""

import cuda.tile as ct
import torch

TILE_SIZE = 32
TILE_M = 16
TILE_N = 16


@ct.kernel
def copy_kernel(src, dst):
    bid = ct.bid(0)
    tile = ct.load(src, index=(bid,), shape=(TILE_SIZE,))
    ct.store(dst, index=(bid,), tile=tile)


@ct.kernel
def padded_scale_kernel(src, dst, scale: float):
    bid = ct.bid(0)
    tile = ct.load(src, index=(bid,), shape=(TILE_SIZE,),
                   padding_mode=ct.PaddingMode.ZERO)
    out_tile = tile * scale
    ct.store(dst, index=(bid,), tile=out_tile)


@ct.kernel
def copy_2d_kernel(src2d, dst2d):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    tile = ct.load(src2d, index=(bid_m, bid_n), shape=(TILE_M, TILE_N))
    ct.store(dst2d, index=(bid_m, bid_n), tile=tile)


@ct.kernel
def downsample_even_kernel(src, dst):
    bid = ct.bid(0)
    tile = ct.load(src, index=(bid,), shape=(TILE_SIZE,))
    # Keep this exercise focused on load/store; output equals input tile.
    ct.store(dst, index=(bid,), tile=tile)


def launch_copy(src: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(src)
    grid = ((src.shape[0] + TILE_SIZE - 1) // TILE_SIZE, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, copy_kernel, (src, out))
    return out


def launch_padded_scale(src: torch.Tensor, scale: float) -> torch.Tensor:
    out = torch.empty_like(src)
    grid = ((src.shape[0] + TILE_SIZE - 1) // TILE_SIZE, 1, 1)
    ct.launch(torch.cuda.current_stream(), grid, padded_scale_kernel, (src, out, scale))
    return out


def launch_copy_2d(src2d: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(src2d)
    grid_m = (src2d.shape[0] + TILE_M - 1) // TILE_M
    grid_n = (src2d.shape[1] + TILE_N - 1) // TILE_N
    ct.launch(torch.cuda.current_stream(), (grid_m, grid_n, 1), copy_2d_kernel, (src2d, out))
    return out


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 03 solution.")
