"""
Module 02 — Tensors
Exercise 01 — GMEM Tensor Creation

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto gmem_tensor = make_tensor(make_gmem_ptr(ptr), make_shape(M, N), stride=(N, 1));
  DSL:  gmem_tensor = cute.make_gmem_tensor(ptr, shape=(M, N), stride=(N, 1))
  Key:  GMEM tensors wrap a CUDA pointer with a layout for indexed access.

WHAT YOU'RE BUILDING:
  A global memory tensor representing a matrix in device memory. This is the
  standard I/O format for kernels — all inputs/outputs live in GMEM initially.
  You'll create the tensor, write values, and verify the layout mapping.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create GMEM tensors from PyTorch tensors via from_dlpack
  - Understand the relationship between pointer, shape, stride, and layout
  - Read/write elements using logical coordinates

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tensor.html
  - from_dlpack: https://nvidia.github.io/cutlass-dsl/cute/dlpack.html
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: If we create a GMEM tensor with shape (64, 32) and stride (32, 1),
#     what is the linear index of element (10, 5)?
# Your answer: 10 * 32 + 5 = 325

# Q2: What is the total memory size in elements for this tensor?
# Your answer: 64 * 32 = 2048

# Q3: How does from_dlpack enable zero-copy interop with PyTorch?
# Your answer: DLPack is a standard tensor format. from_dlpack wraps the
#              existing CUDA pointer without copying data.


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
M, N = 64, 32

# Allocate a PyTorch tensor in CUDA memory
torch_tensor = torch.arange(M * N, dtype=torch.float32, device='cuda').reshape(M, N)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_gmem_tensor(
    tensor_gmem: cute.Tensor,
    results: cute.Tensor,
):
    """
    Create a GMEM tensor view and query its properties.
    
    FILL IN [EASY]: Use from_dlpack to get the GMEM tensor and access elements.
    
    HINT: The input tensor_gmem is already a GMEM tensor from the host.
          Access elements using tensor(coord) for reading.
    """
    # --- Step 1: Verify tensor properties ---
    results[0] = tensor_gmem.shape[0]
    results[1] = tensor_gmem.shape[1]
    
    # --- Step 2: Read element at (10, 5) ---
    results[2] = tensor_gmem((10, 5))
    
    # --- Step 3: Read element at (32, 16) ---
    results[3] = tensor_gmem((32, 16))
    
    # --- Step 4: Compute expected value based on initialization ---
    results[4] = 10 * 32 + 5  # = 325
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify GMEM tensor access.
    
    NCU PROFILING COMMAND:
    ncu --set full --target-processes all python ex01_gmem_tensor_FILL_IN.py
    """
    
    # Convert PyTorch tensor to CuTe GMEM tensor
    gmem_tensor = from_dlpack(torch_tensor)
    
    # Allocate results tensor (5 float32 values)
    results_torch = torch.zeros(5, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel
    kernel_gmem_tensor(gmem_tensor, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 02 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  GMEM Tensor: shape=({M}, {N}), stride=({N}, 1)")
    print(f"\n  Results:")
    print(f"    Shape[0] (M):           {results_cpu[0]}")
    print(f"    Shape[1] (N):           {results_cpu[1]}")
    print(f"    Element (10, 5):        {results_cpu[2]}")
    print(f"    Element (32, 16):       {results_cpu[3]}")
    print(f"    Expected (10, 5):       {results_cpu[4]}")
    
    # Verify
    expected_shape_0 = M
    expected_shape_1 = N
    expected_elem_10_5 = 10 * N + 5  # = 325
    expected_elem_32_16 = 32 * N + 16  # = 1040
    expected_check = 325
    
    passed = (
        results_cpu[0] == expected_shape_0 and
        results_cpu[1] == expected_shape_1 and
        results_cpu[2] == expected_elem_10_5 and
        results_cpu[3] == expected_elem_32_16 and
        results_cpu[4] == expected_check
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: Did your PREDICT answers match?
# C2: How does GMEM tensor access differ from raw pointer arithmetic?
# C3: In a GEMM kernel, which matrices (A, B, C) would be GMEM tensors?
# C4: What is the latency cost of GMEM access vs SMEM?

if __name__ == "__main__":
    run()
