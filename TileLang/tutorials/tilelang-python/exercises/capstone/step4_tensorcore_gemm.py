"""Capstone Step 4: Tensor Core GEMM.

Replace inner math path with Tensor Core instructions.
"""

# TODO: import tilelang + tensorcore/WMMA APIs

BM, BN, BK = 128, 128, 32
WM, WN, WK = 16, 16, 16


def build_tensorcore_gemm(m: int, n: int, k: int):
    # TODO:
    # 1) Define fragment shapes using WM/WN/WK
    # 2) Load matrix fragments
    # 3) MMA accumulate
    # 4) Store accumulated fragments
    raise NotImplementedError
