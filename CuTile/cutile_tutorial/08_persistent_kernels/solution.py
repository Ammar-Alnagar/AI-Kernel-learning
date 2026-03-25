# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 08: Persistent Kernels - Solution"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
TILE_SIZE = 64


def compute_num_tiles(n: int, tile_size: int) -> int:
    return (n + tile_size - 1) // tile_size


@ct.kernel
def persistent_scale_kernel(x, y,
                            num_tiles: ConstInt,
                            total_blocks: ConstInt,
                            scale: float):
    bid = ct.bid(0)

    for t in range(bid, num_tiles, total_blocks):
        tile = ct.load(x, index=(t,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
        out_tile = tile * scale
        ct.store(y, index=(t,), tile=out_tile)


@ct.kernel
def persistent_affine_kernel(x, y,
                             num_tiles: ConstInt,
                             total_blocks: ConstInt,
                             alpha: float,
                             beta: float):
    bid = ct.bid(0)

    for t in range(bid, num_tiles, total_blocks):
        tile = ct.load(x, index=(t,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
        out_tile = tile * alpha + beta
        ct.store(y, index=(t,), tile=out_tile)


def launch_persistent_scale(x: torch.Tensor, scale: float,
                            launch_blocks: int) -> torch.Tensor:
    y = torch.empty_like(x)
    num_tiles = compute_num_tiles(x.shape[0], TILE_SIZE)
    grid = (launch_blocks, 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, persistent_scale_kernel,
              (x, y, num_tiles, launch_blocks, scale))
    return y


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 08 solution.")
