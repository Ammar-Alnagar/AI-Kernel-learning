# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Module 05: Matrix Operations - Test Suite"""

import sys
import torch

try:
    from kernel import launch_tiled_matmul
except ImportError as e:
    print(f"ERROR: Cannot import kernel.py: {e}")
    sys.exit(1)


def test_tiled_matmul_fp32():
    a = torch.randn(96, 128, device="cuda", dtype=torch.float32)
    b = torch.randn(128, 64, device="cuda", dtype=torch.float32)

    c = launch_tiled_matmul(a, b)
    ref = a @ b

    ok = torch.allclose(c, ref, rtol=1e-4, atol=1e-4)
    print(f"tiled matmul fp32: {'✓' if ok else '✗'}")
    return ok


def test_tiled_matmul_non_multiple():
    a = torch.randn(70, 90, device="cuda", dtype=torch.float32)
    b = torch.randn(90, 45, device="cuda", dtype=torch.float32)

    c = launch_tiled_matmul(a, b)
    ref = a @ b

    ok = torch.allclose(c, ref, rtol=1e-4, atol=1e-4)
    print(f"tiled matmul non-multiple: {'✓' if ok else '✗'}")
    return ok


def run_all_tests():
    print("=" * 60)
    print("Module 05: Comprehensive Test Suite")
    print("=" * 60)
    tests = [test_tiled_matmul_fp32, test_tiled_matmul_non_multiple]
    passed = sum(1 for t in tests if t())
    failed = len(tests) - passed
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    return failed == 0


if __name__ == "__main__":
    sys.exit(0 if run_all_tests() else 1)
