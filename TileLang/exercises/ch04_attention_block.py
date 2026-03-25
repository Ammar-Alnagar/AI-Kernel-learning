"""Chapter 4 Exercise: Attention-style block kernel.

Implement a simplified QK^T block multiply path with Tensor Core primitives.
"""

# TODO: import tilelang


def build_qk_block_kernel(seq: int, dim: int):
    # TODO:
    # 1) Tile sequence and head dimension
    # 2) Compute block of QK^T
    # 3) Apply optional scaling
    # 4) Write output score block
    raise NotImplementedError
