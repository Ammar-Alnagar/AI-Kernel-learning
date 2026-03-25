# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 07: Reductions and Atomics - Solution"""

import cuda.tile as ct
import torch

TILE_SIZE = 64


@ct.kernel
def block_sum_kernel(x, block_sums):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    s = ct.sum(x_tile)
    block_sums[bid] = s


@ct.kernel
def block_max_kernel(x, block_maxes):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    m = ct.max(x_tile)
    block_maxes[bid] = m


@ct.kernel
def partial_accumulate_kernel(x, partial_sums):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE_SIZE,), padding_mode=ct.PaddingMode.ZERO)
    partial = ct.sum(x_tile)
    partial_sums[bid] = partial


def launch_block_sum(x: torch.Tensor) -> torch.Tensor:
    num_blocks = (x.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    block_sums = torch.zeros(num_blocks, dtype=x.dtype, device=x.device)
    ct.launch(torch.cuda.current_stream(), (num_blocks, 1, 1), block_sum_kernel, (x, block_sums))
    return block_sums


def launch_block_max(x: torch.Tensor) -> torch.Tensor:
    num_blocks = (x.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    block_maxes = torch.empty(num_blocks, dtype=x.dtype, device=x.device)
    ct.launch(torch.cuda.current_stream(), (num_blocks, 1, 1), block_max_kernel, (x, block_maxes))
    return block_maxes


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 07 solution.")
