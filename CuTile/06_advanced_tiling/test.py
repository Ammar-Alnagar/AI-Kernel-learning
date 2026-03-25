# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 06: Advanced Tiling - Test Suite"""

import sys
import torch

try:
    from kernel import xor_swizzle, launch_map_2d
except ImportError as e:
    print(f"ERROR: Cannot import kernel.py: {e}")
    sys.exit(1)


def test_xor_swizzle():
    cases = [(0, 1, 1), (5, 3, 6), (7, 7, 0)]
    ok = True
    for tile_id, mask, expected in cases:
        got = xor_swizzle(tile_id, mask)
        if got != expected:
            ok = False
    print(f"xor swizzle helper: {'✓' if ok else '✗'}")
    return ok


def test_map_2d():
    x = torch.randn(65, 47, device="cuda", dtype=torch.float32)
    y = launch_map_2d(x)
    ok = torch.allclose(y, x * 2.0, rtol=1e-5, atol=1e-7)
    print(f"2d map kernel: {'✓' if ok else '✗'}")
    return ok


def run_all_tests():
    print("=" * 60)
    print("Module 06: Comprehensive Test Suite")
    print("=" * 60)
    tests = [test_xor_swizzle, test_map_2d]
    passed = sum(1 for t in tests if t())
    failed = len(tests) - passed
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
