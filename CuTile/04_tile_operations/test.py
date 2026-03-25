# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 04: Tile Operations - Test Suite"""

import sys
import torch
import cuda.tile as ct

try:
    from kernel import (
        TILE_SIZE, TILE_M, TILE_N,
        affine_kernel, scalar_broadcast_add_kernel,
        launch_affine, launch_scalar_add,
    )
except ImportError as e:
    print(f"ERROR: Cannot import kernel.py: {e}")
    sys.exit(1)


def test_affine():
    x = torch.randn(513, device="cuda", dtype=torch.float32)
    y = launch_affine(x, 1.25, -0.5)
    ok = torch.allclose(y, x * 1.25 - 0.5, rtol=1e-5, atol=1e-7)
    print(f"affine: {'✓' if ok else '✗'}")
    return ok


def test_scalar_add():
    x = torch.randn(199, device="cuda", dtype=torch.float32)
    y = launch_scalar_add(x, 3.0)
    ok = torch.allclose(y, x + 3.0, rtol=1e-5, atol=1e-7)
    print(f"scalar add: {'✓' if ok else '✗'}")
    return ok


def run_all_tests():
    print("=" * 60)
    print("Module 04: Comprehensive Test Suite")
    print("=" * 60)
    tests = [test_affine, test_scalar_add]
    passed = sum(1 for t in tests if t())
    failed = len(tests) - passed
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
