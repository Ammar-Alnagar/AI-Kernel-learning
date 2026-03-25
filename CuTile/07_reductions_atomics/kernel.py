# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 07: Reductions and Atomics - Fill-in Code Exercise"""

import cuda.tile as ct
import torch

TILE_SIZE = 64


@ct.kernel
def block_sum_kernel(x, block_sums):
    """Compute one sum per tile and store into block_sums."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN

    # FILL IN
    s =  # FILL IN

    # FILL IN: store scalar to block_sums[bid]
    # FILL IN


@ct.kernel
def block_max_kernel(x, block_maxes):
    """Compute one max per tile and store into block_maxes."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN

    # FILL IN
    m =  # FILL IN

    # FILL IN
    # FILL IN


@ct.kernel
def partial_accumulate_kernel(x, partial_sums):
    """Write one partial sum per block (two-pass reduction pattern)."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN
    x_tile =  # FILL IN

    # FILL IN
    partial =  # FILL IN

    # FILL IN: write partial into partial_sums[bid]
    # FILL IN


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
    print("Run `python test.py` to validate Module 07.")
