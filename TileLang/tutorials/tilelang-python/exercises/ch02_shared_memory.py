"""Chapter 2 Exercise: Shared-memory staging.

Stage data through shared memory before writing output.
"""

# TODO: import tilelang

BLOCK = 256


def build_shared_mem_kernel(n: int):
    # TODO:
    # 1) Allocate shared-memory tile/buffer
    # 2) Cooperative load from global to shared
    # 3) Synchronize threads
    # 4) Consume shared data and write output
    raise NotImplementedError
