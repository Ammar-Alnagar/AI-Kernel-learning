# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 06: Advanced Tiling - Solution"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

TILE_M = 16
TILE_N = 16
TILE_SIZE = 32


def xor_swizzle(tile_id: int, mask: int) -> int:
    return tile_id ^ mask


@ct.kernel
def map_2d_kernel(x2d, y2d):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    x_tile = ct.load(x2d, index=(bid_m, bid_n), shape=(TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    y_tile = x_tile * 2.0

    ct.store(y2d, index=(bid_m, bid_n), tile=y_tile)


@ct.kernel
def map_3d_batch_kernel(x3d, y3d, scale: float):
    bid_b = ct.bid(0)
    bid_m = ct.bid(1)
    bid_n = ct.bid(2)

    x_tile = ct.load(x3d, index=(bid_b, bid_m, bid_n), shape=(1, TILE_M, TILE_N), padding_mode=ct.PaddingMode.ZERO)
    y_tile = x_tile * scale

    ct.store(y3d, index=(bid_b, bid_m, bid_n), tile=y_tile)


@ct.kernel
def swizzled_1d_kernel(x, y, mask: ConstInt):
    bid = ct.bid(0)
    swizzled_id = bid ^ mask
    tile = ct.load(x, index=(swizzled_id,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    ct.store(y, index=(bid,), tile=tile)


def launch_map_2d(x2d: torch.Tensor) -> torch.Tensor:
    y2d = torch.empty_like(x2d)
    grid = ((x2d.shape[0] + TILE_M - 1) // TILE_M,
            (x2d.shape[1] + TILE_N - 1) // TILE_N,
            1)
    ct.launch(torch.cuda.current_stream(), grid, map_2d_kernel, (x2d, y2d))
    return y2d


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 06 solution.")
