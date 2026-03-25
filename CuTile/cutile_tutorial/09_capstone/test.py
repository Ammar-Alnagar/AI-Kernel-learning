# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 09: Capstone FMHA - Test Suite"""

import sys
import math
import torch

try:
    from kernel import fmha_forward, normalize_scores_inplace
except ImportError as e:
    print(f"ERROR: Cannot import kernel.py: {e}")
    sys.exit(1)


def reference_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(d)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def test_softmax_normalization():
    x = torch.randn(2, 3, 16, 16, device="cuda", dtype=torch.float32)
    p = normalize_scores_inplace(x)
    row_sums = p.sum(dim=-1)
    ones = torch.ones_like(row_sums)
    ok = torch.allclose(row_sums, ones, rtol=1e-5, atol=1e-6)
    print(f"row normalization: {'✓' if ok else '✗'}")
    return ok


def test_fmha_forward_small():
    q = torch.randn(2, 2, 64, 32, device="cuda", dtype=torch.float32)
    k = torch.randn(2, 2, 64, 32, device="cuda", dtype=torch.float32)
    v = torch.randn(2, 2, 64, 32, device="cuda", dtype=torch.float32)

    y = fmha_forward(q, k, v)
    ref = reference_attention(q, k, v)

    ok = torch.allclose(y, ref, rtol=5e-3, atol=5e-3)
    print(f"fmha forward small: {'✓' if ok else '✗'}")
    return ok


def run_all_tests():
    print("=" * 60)
    print("Module 09: Capstone FMHA Test Suite")
    print("=" * 60)
    tests = [test_softmax_normalization, test_fmha_forward_small]
    passed = sum(1 for t in tests if t())
    failed = len(tests) - passed
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
