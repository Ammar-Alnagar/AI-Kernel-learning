# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 04: Tile Operations - Solution"""

import cuda.tile as ct
import torch

TILE_SIZE = 32
TILE_M = 16
TILE_N = 16


@ct.kernel
def affine_kernel(x, y, alpha: float, beta: float):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    y_tile = x_tile * alpha + beta
    ct.store(y, index=(bid,), tile=y_tile)


@ct.kernel
def scalar_broadcast_add_kernel(x, y, scalar: float):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    y_tile = x_tile + scalar
    ct.store(y, index=(bid,), tile=y_tile)


@ct.kernel
def transpose_2d_kernel(x2d, y2d):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)
    x_tile = ct.load(x2d, index=(bid_m, bid_n), shape=(TILE_M, TILE_N))
    y_tile = ct.transpose(x_tile)
    ct.store(y2d, index=(bid_n, bid_m), tile=y_tile)


@ct.kernel
def tile_center_kernel(x, y):
    bid = ct.bid(0)
    x_tile = ct.load(x, index=(bid,), shape=(TILE_SIZE,))
    mean_val = ct.sum(x_tile) / TILE_SIZE
    y_tile = x_tile - mean_val
    ct.store(y, index=(bid,), tile=y_tile)


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
    print("Run `python test.py` to validate Module 04 solution.")
