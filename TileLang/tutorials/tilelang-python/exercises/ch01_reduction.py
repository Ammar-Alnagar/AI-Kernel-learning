"""Chapter 1 Exercise: Row-wise reduction.

Implement a kernel that sums each row of a 2D tensor.
"""

# TODO: import tilelang


def build_row_sum_kernel(m: int, n: int):
    # TODO:
    # 1) Assign one program per row (or tiled rows)
    # 2) Accumulate along columns
    # 3) Write output vector of shape [m]
    raise NotImplementedError


def run_and_check(x, y_ref):
    # TODO: run kernel and return error statistics
    raise NotImplementedError


if __name__ == "__main__":
    print("Fill TODOs for row-wise reduction kernel.")
