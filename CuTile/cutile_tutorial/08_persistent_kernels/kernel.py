# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 08: Persistent Kernels - Fill-in Code Exercise"""

import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]
TILE_SIZE = 64


def compute_num_tiles(n: int, tile_size: int) -> int:
    """Host helper: ceil division for number of tiles."""
    # FILL IN
    return  # FILL IN


@ct.kernel
def persistent_scale_kernel(x, y,
                            num_tiles: ConstInt,
                            total_blocks: ConstInt,
                            scale: float):
    """Each block processes multiple tiles via strided tile loop."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN: iterate t = bid, bid + total_blocks, ... < num_tiles
    # for t in range(...):
    #   tile = load(x, t)
    #   tile = tile * scale
    #   store(y, t)


@ct.kernel
def persistent_affine_kernel(x, y,
                             num_tiles: ConstInt,
                             total_blocks: ConstInt,
                             alpha: float,
                             beta: float):
    """Persistent affine transform: y = alpha*x + beta."""
    # FILL IN
    bid =  # FILL IN

    # FILL IN loop pattern (same as above)


def launch_persistent_scale(x: torch.Tensor, scale: float,
                            launch_blocks: int) -> torch.Tensor:
    y = torch.empty_like(x)
    num_tiles = compute_num_tiles(x.shape[0], TILE_SIZE)
    grid = (launch_blocks, 1, 1)

    ct.launch(torch.cuda.current_stream(), grid, persistent_scale_kernel,
              (x, y, num_tiles, launch_blocks, scale))
    return y


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 08.")
