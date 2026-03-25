# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 05: Matrix Operations - Solution"""

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
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    a_tile = ct.load(a, index=(bid_m, 0), shape=(tile_m, tile_k), padding_mode=ct.PaddingMode.ZERO)
    b_tile = ct.load(b, index=(0, bid_n), shape=(tile_k, tile_n), padding_mode=ct.PaddingMode.ZERO)

    c_tile = ct.matmul(a_tile, b_tile)
    ct.store(c, index=(bid_m, bid_n), tile=c_tile)


@ct.kernel
def tiled_matmul_kloop_kernel(a, b, c,
                              tile_m: ConstInt,
                              tile_n: ConstInt,
                              tile_k: ConstInt,
                              k_tiles: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.zeros((tile_m, tile_n))

    for k_id in range(k_tiles):
        a_tile = ct.load(a, index=(bid_m, k_id), shape=(tile_m, tile_k), padding_mode=ct.PaddingMode.ZERO)
        b_tile = ct.load(b, index=(k_id, bid_n), shape=(tile_k, tile_n), padding_mode=ct.PaddingMode.ZERO)
        acc = acc + ct.matmul(a_tile, b_tile)

    ct.store(c, index=(bid_m, bid_n), tile=acc)


@ct.kernel
def mma_style_kernel(a, b, c,
                     tile_m: ConstInt,
                     tile_n: ConstInt,
                     tile_k: ConstInt,
                     k_tiles: ConstInt):
    bid_m = ct.bid(0)
    bid_n = ct.bid(1)

    acc = ct.zeros((tile_m, tile_n))

    for k_id in range(k_tiles):
        a_frag = ct.load(a, index=(bid_m, k_id), shape=(tile_m, tile_k), padding_mode=ct.PaddingMode.ZERO)
        b_frag = ct.load(b, index=(k_id, bid_n), shape=(tile_k, tile_n), padding_mode=ct.PaddingMode.ZERO)
        acc = ct.mma(a_frag, b_frag, acc)

    ct.store(c, index=(bid_m, bid_n), tile=acc)


def launch_tiled_matmul(a: torch.Tensor, b: torch.Tensor,
                        tile_m: int = TILE_M,
                        tile_n: int = TILE_N,
                        tile_k: int = TILE_K) -> torch.Tensor:
    if a.shape[1] != b.shape[0]:
        raise ValueError("Inner dimensions must match")

    m, k = a.shape
    _, n = b.shape

    c = torch.empty((m, n), dtype=a.dtype, device=a.device)

    grid_m = (m + tile_m - 1) // tile_m
    grid_n = (n + tile_n - 1) // tile_n
    k_tiles = (k + tile_k - 1) // tile_k

    grid = (grid_m, grid_n, 1)
    ct.launch(torch.cuda.current_stream(), grid, tiled_matmul_kloop_kernel,
              (a, b, c, tile_m, tile_n, tile_k, k_tiles))

    return c


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 05 solution.")
