"""Chapter 3 Exercise: Memory coalescing optimization.

Start from a deliberately poor access pattern and fix it.
"""

# TODO: import tilelang


def build_uncoalesced_kernel(n: int):
    # TODO: implement a baseline with suboptimal access pattern
    raise NotImplementedError


def build_coalesced_kernel(n: int):
    # TODO: implement improved contiguous access pattern
    raise NotImplementedError


def benchmark_pair(x):
    # TODO: benchmark both kernels, return latencies and speedup
    raise NotImplementedError
