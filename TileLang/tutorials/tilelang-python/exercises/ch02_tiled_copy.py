"""Chapter 2 Exercise: Tiled copy kernel.

Goal: copy X -> Y using explicit tiling and boundary guards.
"""

# TODO: import tilelang

TILE_M = 128
TILE_N = 128


def build_tiled_copy_kernel(m: int, n: int):
    # TODO:
    # 1) Map program IDs to tile coordinates
    # 2) Compute tile-local indices
    # 3) Guard partial tiles at boundaries
    # 4) Store tile to output
    raise NotImplementedError
