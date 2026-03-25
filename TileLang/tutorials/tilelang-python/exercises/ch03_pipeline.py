"""Chapter 3 Exercise: Software pipeline stages.

Implement load/compute overlap strategy across tiles.
"""

# TODO: import tilelang

PIPE_STAGES = 2


def build_pipelined_kernel(m: int, n: int, k: int):
    # TODO:
    # 1) Create loop over K tiles
    # 2) Prefetch next tile while computing current tile
    # 3) Use stage buffers correctly
    # 4) Verify no data hazards
    raise NotImplementedError
