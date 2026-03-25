# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 08: Persistent Kernels - Test Suite"""

import sys
import torch

try:
    from kernel import TILE_SIZE, compute_num_tiles, launch_persistent_scale
except ImportError as e:
    print(f"ERROR: Cannot import kernel.py: {e}")
    sys.exit(1)


def test_num_tiles():
    cases = [(1, TILE_SIZE, 1), (64, TILE_SIZE, 1), (65, TILE_SIZE, 2), (256, TILE_SIZE, 4)]
    ok = True
    for n, ts, expected in cases:
        got = compute_num_tiles(n, ts)
        if got != expected:
            ok = False
    print(f"compute_num_tiles: {'✓' if ok else '✗'}")
    return ok


def test_persistent_scale():
    x = torch.randn(4097, device="cuda", dtype=torch.float32)
    y = launch_persistent_scale(x, 0.75, launch_blocks=8)
    ok = torch.allclose(y, x * 0.75, rtol=1e-5, atol=1e-7)
    print(f"persistent scale kernel: {'✓' if ok else '✗'}")
    return ok


def run_all_tests():
    print("=" * 60)
    print("Module 08: Comprehensive Test Suite")
    print("=" * 60)
    tests = [test_num_tiles, test_persistent_scale]
    passed = sum(1 for t in tests if t())
    failed = len(tests) - passed
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
