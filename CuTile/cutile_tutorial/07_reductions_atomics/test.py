# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 07: Reductions and Atomics - Test Suite"""

import sys
import torch

try:
    from kernel import TILE_SIZE, launch_block_sum, launch_block_max
except ImportError as e:
    print(f"ERROR: Cannot import kernel.py: {e}")
    sys.exit(1)


def test_block_sum_reduction():
    x = torch.randn(1000, device="cuda", dtype=torch.float32)
    block_sums = launch_block_sum(x)
    total = block_sums.sum()
    ref = x.sum()
    ok = torch.allclose(total, ref, rtol=1e-4, atol=1e-4)
    print(f"block sum reduction: {'✓' if ok else '✗'}")
    return ok


def test_block_max_reduction():
    x = torch.randn(1000, device="cuda", dtype=torch.float32)
    block_maxes = launch_block_max(x)
    got = block_maxes.max()
    ref = x.max()
    ok = torch.allclose(got, ref, rtol=1e-5, atol=1e-6)
    print(f"block max reduction: {'✓' if ok else '✗'}")
    return ok


def run_all_tests():
    print("=" * 60)
    print("Module 07: Comprehensive Test Suite")
    print("=" * 60)
    print(f"TILE_SIZE={TILE_SIZE}")
    tests = [test_block_sum_reduction, test_block_max_reduction]
    passed = sum(1 for t in tests if t())
    failed = len(tests) - passed
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
