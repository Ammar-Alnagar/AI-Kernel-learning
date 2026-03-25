# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 09: Capstone FMHA - Fill-in Code Exercise"""

import math
import cuda.tile as ct
import torch

ConstInt = ct.Constant[int]

TILE_S = 32
TILE_D = 32


@ct.kernel
def score_kernel(q, k, scores,
                 tile_s: ConstInt,
                 tile_d: ConstInt,
                 q_tiles: ConstInt,
                 k_tiles: ConstInt,
                 scale: float):
    """Compute one score tile for fixed (batch, head, query_tile, key_tile)."""
    # Grid mapping:
    # dim0 -> batch
    # dim1 -> head
    # dim2 -> flattened (query_tile, key_tile)

    # FILL IN
    bid_b =  # FILL IN
    bid_h =  # FILL IN
    bid_qk =  # FILL IN

    # FILL IN: derive q_tile_id and k_tile_id from bid_qk
    q_tile_id =  # FILL IN
    k_tile_id =  # FILL IN

    # FILL IN: load q tile shape (1, 1, tile_s, tile_d)
    q_tile =  # FILL IN

    # FILL IN: load k tile shape (1, 1, tile_s, tile_d)
    k_tile =  # FILL IN

    # FILL IN: transpose K tile to (tile_d, tile_s)
    k_t =  # FILL IN

    # FILL IN: compute scaled score tile
    s_tile =  # FILL IN

    # FILL IN: store score tile into scores
    # FILL IN


@ct.kernel
def pv_kernel(probs, v, out,
              tile_s: ConstInt,
              tile_d: ConstInt,
              k_tiles: ConstInt):
    """Compute one output tile: out = probs @ v."""
    # FILL IN
    bid_b =  # FILL IN
    bid_h =  # FILL IN
    bid_q =  # FILL IN

    # FILL IN: initialize output accumulator tile
    out_acc =  # FILL IN

    # FILL IN: loop over key tiles and accumulate
    # for k_tile_id in range(...):
    #   p_tile = ...
    #   v_tile = ...
    #   out_acc = out_acc + ct.matmul(p_tile, v_tile)

    # FILL IN: store out_acc
    # FILL IN


def normalize_scores_inplace(scores: torch.Tensor) -> torch.Tensor:
    """Host-side softmax normalization along last dimension for clarity."""
    # FILL IN: implement stable softmax with torch operations
    # hint: max-subtract, exp, sum
    return  # FILL IN


def fmha_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Tutorial FMHA forward pass using tiled CuTile kernels + host normalization."""
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, V must have the same shape")

    bsz, heads, seqlen, dim = q.shape
    if dim % TILE_D != 0:
        raise ValueError("Head dimension must be divisible by TILE_D")

    scale = 1.0 / math.sqrt(dim)

    scores = torch.empty((bsz, heads, seqlen, seqlen), device=q.device, dtype=q.dtype)
    out = torch.empty_like(q)

    # Launch score kernel
    q_tiles = (seqlen + TILE_S - 1) // TILE_S
    k_tiles = (seqlen + TILE_S - 1) // TILE_S
    grid_scores = (bsz, heads, q_tiles * k_tiles)

    # FILL IN: ct.launch for score_kernel
    # Pass q_tiles and k_tiles as compile-time constants
    # FILL IN

    probs = normalize_scores_inplace(scores)

    # Launch P@V kernel (one query tile per block along dim2)
    grid_pv = (bsz, heads, q_tiles)

    # FILL IN: ct.launch for pv_kernel
    # Pass k_tiles as a compile-time constant
    # FILL IN

    return out


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 09.")
