# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 09: Capstone FMHA - Solution"""

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
    bid_b = ct.bid(0)
    bid_h = ct.bid(1)
    bid_qk = ct.bid(2)

    q_tile_id = bid_qk // k_tiles
    k_tile_id = bid_qk % k_tiles

    q_tile = ct.load(q, index=(bid_b, bid_h, q_tile_id, 0),
                     shape=(1, 1, tile_s, tile_d), padding_mode=ct.PaddingMode.ZERO)
    k_tile = ct.load(k, index=(bid_b, bid_h, k_tile_id, 0),
                     shape=(1, 1, tile_s, tile_d), padding_mode=ct.PaddingMode.ZERO)

    q2 = ct.reshape(q_tile, (tile_s, tile_d))
    k2 = ct.reshape(k_tile, (tile_s, tile_d))
    k_t = ct.transpose(k2)

    s_tile = ct.matmul(q2, k_t) * scale
    s_tile4 = ct.reshape(s_tile, (1, 1, tile_s, tile_s))

    ct.store(scores, index=(bid_b, bid_h, q_tile_id, k_tile_id), tile=s_tile4)


@ct.kernel
def pv_kernel(probs, v, out,
              tile_s: ConstInt,
              tile_d: ConstInt,
              k_tiles: ConstInt):
    bid_b = ct.bid(0)
    bid_h = ct.bid(1)
    bid_q = ct.bid(2)

    out_acc = ct.zeros((tile_s, tile_d))

    for k_tile_id in range(k_tiles):
        p_tile4 = ct.load(probs, index=(bid_b, bid_h, bid_q, k_tile_id),
                          shape=(1, 1, tile_s, tile_s), padding_mode=ct.PaddingMode.ZERO)
        v_tile4 = ct.load(v, index=(bid_b, bid_h, k_tile_id, 0),
                          shape=(1, 1, tile_s, tile_d), padding_mode=ct.PaddingMode.ZERO)

        p_tile = ct.reshape(p_tile4, (tile_s, tile_s))
        v_tile = ct.reshape(v_tile4, (tile_s, tile_d))

        out_acc = out_acc + ct.matmul(p_tile, v_tile)

    out_tile4 = ct.reshape(out_acc, (1, 1, tile_s, tile_d))
    ct.store(out, index=(bid_b, bid_h, bid_q, 0), tile=out_tile4)


def normalize_scores_inplace(scores: torch.Tensor) -> torch.Tensor:
    x = scores
    x_max = x.max(dim=-1, keepdim=True).values
    x_exp = torch.exp(x - x_max)
    x_sum = x_exp.sum(dim=-1, keepdim=True)
    return x_exp / x_sum


def fmha_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("Q, K, V must have the same shape")

    bsz, heads, seqlen, dim = q.shape
    if dim % TILE_D != 0:
        raise ValueError("Head dimension must be divisible by TILE_D")

    scale = 1.0 / math.sqrt(dim)

    scores = torch.empty((bsz, heads, seqlen, seqlen), device=q.device, dtype=q.dtype)
    out = torch.empty_like(q)

    q_tiles = (seqlen + TILE_S - 1) // TILE_S
    k_tiles = (seqlen + TILE_S - 1) // TILE_S
    grid_scores = (bsz, heads, q_tiles * k_tiles)

    ct.launch(torch.cuda.current_stream(), grid_scores, score_kernel,
              (q, k, scores, TILE_S, TILE_D, q_tiles, k_tiles, scale))

    probs = normalize_scores_inplace(scores)

    grid_pv = (bsz, heads, q_tiles)
    ct.launch(torch.cuda.current_stream(), grid_pv, pv_kernel,
              (probs, v, out, TILE_S, TILE_D, k_tiles))

    return out


if __name__ == "__main__":
    print("Run `python test.py` to validate Module 09 solution.")
