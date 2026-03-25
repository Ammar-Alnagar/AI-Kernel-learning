# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 03: Load/Store - Test Suite"""

import sys
import torch
import cuda.tile as ct

try:
    from kernel import (
        TILE_SIZE, TILE_M, TILE_N,
        copy_kernel, padded_scale_kernel, copy_2d_kernel,
        launch_copy, launch_padded_scale, launch_copy_2d,
    )
except ImportError as e:
    print(f"ERROR: Cannot import kernel.py: {e}")
    sys.exit(1)


def test_copy_kernel():
    x = torch.randn(257, device="cuda", dtype=torch.float32)
    y = launch_copy(x)
    ok = torch.allclose(x, y, rtol=1e-5, atol=1e-7)
    print(f"copy kernel: {'✓' if ok else '✗'}")
    return ok


def test_padded_scale_kernel():
    x = torch.randn(100, device="cuda", dtype=torch.float32)
    scale = 1.75
    y = launch_padded_scale(x, scale)
    ok = torch.allclose(y, x * scale, rtol=1e-5, atol=1e-7)
    print(f"padded scale: {'✓' if ok else '✗'}")
    return ok


def test_copy_2d_kernel():
    x = torch.randn(70, 34, device="cuda", dtype=torch.float32)
    y = launch_copy_2d(x)
    ok = torch.allclose(x, y, rtol=1e-5, atol=1e-7)
    print(f"2d copy: {'✓' if ok else '✗'}")
    return ok


def run_all_tests():
    print("=" * 60)
    print("Module 03: Comprehensive Test Suite")
    print("=" * 60)
    print(f"TILE_SIZE={TILE_SIZE}, TILE_M={TILE_M}, TILE_N={TILE_N}")

    tests = [test_copy_kernel, test_padded_scale_kernel, test_copy_2d_kernel]
    passed = sum(1 for t in tests if t())
    failed = len(tests) - passed

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
