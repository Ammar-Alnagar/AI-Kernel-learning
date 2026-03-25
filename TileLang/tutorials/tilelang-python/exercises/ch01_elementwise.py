"""Chapter 1 Exercise: Elementwise kernel.

Fill in TODO sections to implement C = A + B.
"""

# TODO: Replace with actual TileLang import(s).
# import tilelang as tl


def build_elementwise_add_kernel(n: int):
    """Return a TileLang kernel/function that computes C = A + B."""
    # TODO:
    # 1) Define kernel signature: (A, B, C)
    # 2) Map program IDs to 1D index
    # 3) Guard out-of-bounds
    # 4) Load A[i], B[i], write C[i]
    raise NotImplementedError


def run_and_check(a, b, c_ref):
    """Compile and run your kernel; compare against c_ref."""
    # TODO: launch kernel and return max absolute error.
    raise NotImplementedError


if __name__ == "__main__":
    print("Complete TODOs in build_elementwise_add_kernel() and run_and_check().")
