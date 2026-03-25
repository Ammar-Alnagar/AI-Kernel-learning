"""Capstone Step 3: Shared-memory GEMM.

Stage A/B tiles in shared memory to reduce global memory traffic.
"""

# TODO: import tilelang

BM, BN, BK = 128, 128, 32


def build_shared_gemm(m: int, n: int, k: int):
    # TODO:
    # 1) Cooperative load A/B tiles into shared memory
    # 2) Synchronize
    # 3) Compute partial products
    # 4) Iterate over K tiles
    raise NotImplementedError
