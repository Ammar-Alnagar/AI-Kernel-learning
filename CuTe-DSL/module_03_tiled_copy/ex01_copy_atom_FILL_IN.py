"""
Module 03 — TiledCopy
Exercise 01 — Copy Atom Basics

CONCEPT BRIDGE (C++ → DSL):
  C++:  using CopyAtom = Copy_Atom<UniversalCopy, float>;
  DSL:  copy_atom = cute.Copy_atom(cute.UniversalCopy, cutlass.float32)
  Key:  Copy atoms define the elementary copy operation (op, src_dtype, dst_dtype).

WHAT YOU'RE BUILDING:
  A copy atom that specifies how data is copied at the element level. Copy atoms
  are the building blocks of TiledCopy — they define vectorization width, memory
  space, and data types. This exercise introduces the atom specification syntax.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Create copy atoms with specified operation and dtypes
  - Understand the role of copy atoms in TiledCopy
  - Distinguish between different copy operations (Universal, Vectorized, TMA)

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tiled_copy.html
  - CuTe C++ mental model: Copy_Atom defines the leaf copy operation
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: What is the difference between UniversalCopy and VectorizedCopy?
# Your answer:

# Q2: For FP16 data, what vectorization width would you choose for maximum
#     throughput on Ampere (assuming 128-bit load instructions)?
# Your answer:

# Q3: Why do copy atoms specify both source and destination dtypes?
# Your answer:


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Copy atom configuration
COPY_OP = cute.UniversalCopy  # Generic copy operation
SRC_DTYPE = cutlass.float32
DST_DTYPE = cutlass.float32


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_copy_atom(
    src_tensor: cute.Tensor,
    dst_tensor: cute.Tensor,
    results: cute.Tensor,
):
    """
    Create and use a copy atom for element-wise copy.
    
    FILL IN [EASY]: Create a copy atom and perform a simple copy.
    
    HINT: copy_atom = cute.Copy_atom(COPY_OP, SRC_DTYPE, DST_DTYPE)
          Then use cute.copy(copy_atom, src, dst) for the actual copy.
    """
    # --- Step 1: Create copy atom ---
    # TODO: copy_atom = cute.Copy_atom(COPY_OP, SRC_DTYPE, DST_DTYPE)
    
    # --- Step 2: Get thread-local slice ---
    # For this simple example, we're using a single thread
    # In real kernels, you'd use tiled_copy.get_slice(thread_idx)
    
    # --- Step 3: Copy elements one at a time using the atom ---
    # TODO: Use cute.copy(copy_atom, src[i], dst[i]) for each element
    # Copy 4 elements from src to dst
    
    # --- Step 4: Verify copy by reading back ---
    # Store dst[0], dst[1], dst[2], dst[3] in results[0:4]
    
    # --- Step 5: Store expected values for verification ---
    # src was initialized with [100, 200, 300, 400, ...]
    # Store expected values in results[4:8]
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify copy atom behavior.
    
    NCU PROFILING COMMAND:
    ncu --set full --target-processes all python ex01_copy_atom_FILL_IN.py
    """
    
    # Create source tensor with known pattern
    src_torch = torch.arange(100, 500, dtype=torch.float32, device='cuda')
    src_cute = from_dlpack(src_torch)
    
    # Create destination tensor (zeros initially)
    dst_torch = torch.zeros(4, dtype=torch.float32, device='cuda')
    dst_cute = from_dlpack(dst_torch)
    
    # Results tensor
    results_torch = torch.zeros(8, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel (1 thread for simplicity)
    kernel_copy_atom[1, 1](src_cute, dst_cute, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    dst_cpu = dst_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 03 — Exercise 01 Results")
    print("=" * 60)
    print(f"\n  Copy Atom: {COPY_OP}, {SRC_DTYPE} → {DST_DTYPE}")
    print(f"\n  Source (first 4): [100, 200, 300, 400]")
    print(f"  Destination (after copy): {dst_cpu}")
    print(f"\n  Results:")
    print(f"    dst[0]: {results_cpu[0]}, expected: {results_cpu[4]}")
    print(f"    dst[1]: {results_cpu[1]}, expected: {results_cpu[5]}")
    print(f"    dst[2]: {results_cpu[2]}, expected: {results_cpu[6]}")
    print(f"    dst[3]: {results_cpu[3]}, expected: {results_cpu[7]}")
    
    # Verify
    passed = (
        results_cpu[0] == results_cpu[4] and
        results_cpu[1] == results_cpu[5] and
        results_cpu[2] == results_cpu[6] and
        results_cpu[3] == results_cpu[7] and
        dst_cpu[0] == 100 and
        dst_cpu[1] == 200 and
        dst_cpu[2] == 300 and
        dst_cpu[3] == 400
    )
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: What is the purpose of a copy atom vs a full TiledCopy?
# C2: When would you use UniversalCopy vs a specialized copy operation?
# C3: How does vectorization improve copy throughput?
# C4: In FlashAttention, which copy atom would you use for loading QKV?

if __name__ == "__main__":
    run()
