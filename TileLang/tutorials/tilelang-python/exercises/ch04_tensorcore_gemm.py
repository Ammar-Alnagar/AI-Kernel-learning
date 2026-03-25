"""Chapter 4 Exercise: Tensor Core GEMM.

Implement Tensor Core path for C = A @ B.
"""

# TODO: import tilelang and tensorcore/WMMA APIs

# Typical starting point; tune for your target GPU.
BLOCK_M = 128
BLOCK_N = 128
BLOCK_K = 32


def build_tensorcore_gemm(m: int, n: int, k: int):
    # TODO:
    # 1) Choose Tensor Core-friendly fragment shapes
    # 2) Load A/B fragments in mixed precision
    # 3) Accumulate in higher precision
    # 4) Store results to C
    raise NotImplementedError


def check_accuracy(c, c_ref):
    # TODO: return max/mean error, relative metrics
    raise NotImplementedError
