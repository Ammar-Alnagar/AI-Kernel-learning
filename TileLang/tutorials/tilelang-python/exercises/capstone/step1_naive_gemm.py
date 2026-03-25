"""Capstone Step 1: Naive GEMM.

Implement straightforward C = A @ B without tiling.
"""

# TODO: import tilelang


def build_naive_gemm(m: int, n: int, k: int):
    # TODO:
    # 1) Map one output element per program/thread
    # 2) Loop over K dimension
    # 3) Accumulate and store C[m, n]
    raise NotImplementedError
