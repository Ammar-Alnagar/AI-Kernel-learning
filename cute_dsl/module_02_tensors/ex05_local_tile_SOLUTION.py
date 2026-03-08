"""
Module 02 — Tensors
Exercise 05 — local_tile for Blocked Access

CONCEPT BRIDGE (C++ → DSL):
  C++:  auto tile = local_tile(tensor, make_tile(M_BLOCK, N_BLOCK), make_coord(m_idx, n_idx));
  DSL:  tile = cute.local_tile(tensor, (M_BLOCK, N_BLOCK), (m_idx, n_idx))
  Key:  local_tile extracts a 2D block from a tensor using tile coordinates.

WHAT YOU'RE BUILDING:
  The core tiling primitive used in FlashAttention and tiled GEMM. `local_tile`
  partitions a large tensor into fixed-size blocks, enabling:
  - Block-wise processing of sequences that don't fit in SMEM
  - Parallel processing across CTAs/warps
  - Cache-efficient memory access patterns
  
  This exercise directly implements the tiling used in FA2's QK^T mainloop.

LEARNING OBJECTIVE:
  After completing this exercise you will be able to:
  - Use local_tile to extract 2D blocks from tensors
  - Understand tile coordinates vs element coordinates
  - Apply tiling to FlashAttention's block-wise attention computation

REQUIRED READING:
  - CUTLASS docs: https://nvidia.github.io/cutlass-dsl/cute/tensor.html#tiling
  - FlashAttention-2 paper: https://arxiv.org/abs/2307.08691 (Section 3 on tiling)
"""

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack
import torch

# ─────────────────────────────────────────────
# PREDICT BEFORE RUNNING
# ─────────────────────────────────────────────
# Q1: For a tensor with shape (128, 64) tiled into (32, 16) blocks,
#     how many tiles are there in each dimension?
# Your answer: 128/32 = 4 tiles in M, 64/16 = 4 tiles in N

# Q2: What is the element coordinate of the top-left corner of tile (2, 3)?
# Your answer: (2*32, 3*16) = (64, 48)

# Q3: In FlashAttention with seq_len=4096 and block_size=64, how many
#     tiles are needed to cover the sequence dimension?
# Your answer: 4096/64 = 64 tiles


# ─────────────────────────────────────────────
# SETUP
# ─────────────────────────────────────────────
# Tensor shape and tile size
M, N = 128, 64
TILE_M, TILE_N = 32, 16

# Number of tiles in each dimension
NUM_TILES_M = M // TILE_M  # = 4
NUM_TILES_N = N // TILE_N  # = 4

# Create tensor with known pattern
torch_tensor = torch.arange(M * N, dtype=torch.float32, device='cuda').reshape(M, N)


# ─────────────────────────────────────────────
# KERNEL
# ─────────────────────────────────────────────
@cutlass.jit
def kernel_local_tile(
    tensor: cute.Tensor,
    results: cute.Tensor,
):
    """
    Demonstrate local_tile for blocked tensor access.
    
    FILL IN [HARD]: Use local_tile to extract blocks and verify contents.
    
    HINT: local_tile(tensor, tile_shape, tile_coord) extracts the tile at
          the given tile coordinate. Element (i, j) in the tile corresponds
          to element (tile_i * TILE_M + i, tile_j * TILE_N + j) in the original.
    """
    # --- Step 1: Extract tile at (2, 3) ---
    tile23 = cute.local_tile(tensor, (TILE_M, TILE_N), (2, 3))
    results[0] = tile23[0, 0]  # Element (64, 48) = 4144
    results[1] = tile23[5, 5]  # Element (69, 53) = 4469
    
    # --- Step 2: Extract tile at (0, 0) ---
    tile00 = cute.local_tile(tensor, (TILE_M, TILE_N), (0, 0))
    results[2] = tile00[0, 0]  # Element (0, 0) = 0
    results[3] = tile00[31, 15]  # Element (31, 15) = 1999
    
    # --- Step 3: Extract tile at (3, 3) ---
    tile33 = cute.local_tile(tensor, (TILE_M, TILE_N), (3, 3))
    results[4] = tile33[0, 0]  # Element (96, 48) = 6192
    
    # --- Step 4: Verify tile shape ---
    results[5] = float(tile23.shape[0])  # = TILE_M = 32
    results[6] = float(tile23.shape[1])  # = TILE_N = 16
    
    # --- Step 5: Iterate over all tiles and sum their top-left elements ---
    total = 0.0
    for ti in range(NUM_TILES_M):
        for tj in range(NUM_TILES_N):
            tile = cute.local_tile(tensor, (TILE_M, TILE_N), (ti, tj))
            total += tile[0, 0]
    results[7] = total
    
    pass


# ─────────────────────────────────────────────
# HOST LAUNCH
# ─────────────────────────────────────────────
def run():
    """
    Run the kernel and verify local_tile behavior.
    
    NCU PROFILING COMMAND:
    ncu --set full --target-processes all python ex05_local_tile_FILL_IN.py
    
    This is the foundation for FlashAttention profiling in later projects.
    """
    
    # Convert to CuTe tensor
    tensor = from_dlpack(torch_tensor)
    
    # Allocate results tensor (8 values)
    results_torch = torch.zeros(8, dtype=torch.float32, device='cuda')
    results_cute = from_dlpack(results_torch)
    
    # Launch kernel
    kernel_local_tile[1, 1](tensor, results_cute)
    
    # Copy back
    results_cpu = results_torch.cpu().numpy()
    
    print("\n" + "=" * 60)
    print("  Module 02 — Exercise 05 Results")
    print("=" * 60)
    print(f"\n  Tensor: shape=({M}, {N})")
    print(f"  Tile size: ({TILE_M}, {TILE_N})")
    print(f"  Grid: ({NUM_TILES_M}, {NUM_TILES_N}) tiles")
    print(f"\n  Results:")
    print(f"    Tile(2,3)[0,0]:     {results_cpu[0]}")
    print(f"    Tile(2,3)[5,5]:     {results_cpu[1]}")
    print(f"    Tile(0,0)[0,0]:     {results_cpu[2]}")
    print(f"    Tile(0,0)[31,15]:   {results_cpu[3]}")
    print(f"    Tile(3,3)[0,0]:     {results_cpu[4]}")
    print(f"    Tile shape[0]:      {results_cpu[5]}")
    print(f"    Tile shape[1]:      {results_cpu[6]}")
    print(f"    Sum of tile[0,0]:   {results_cpu[7]}")
    
    # Verify
    # Tile (2, 3) starts at element (2*32, 3*16) = (64, 48)
    expected_tile23_00 = 64 * 64 + 48  # = 4144
    expected_tile23_55 = (64 + 5) * 64 + (48 + 5)  # = 69*64+53 = 4469
    expected_tile00_00 = 0
    expected_tile00_31_15 = 31 * 64 + 15  # = 1999
    expected_tile33_00 = 96 * 64 + 48  # = 6192
    
    # Sum of all tile[0,0] elements
    expected_sum = 0
    for ti in range(NUM_TILES_M):
        for tj in range(NUM_TILES_N):
            elem_i = ti * TILE_M
            elem_j = tj * TILE_N
            expected_sum += elem_i * N + elem_j
    
    passed = (
        results_cpu[0] == expected_tile23_00 and
        results_cpu[1] == expected_tile23_55 and
        results_cpu[2] == expected_tile00_00 and
        results_cpu[3] == expected_tile00_31_15 and
        results_cpu[4] == expected_tile33_00 and
        results_cpu[5] == TILE_M and
        results_cpu[6] == TILE_N and
        results_cpu[7] == expected_sum
    )
    
    print(f"\n  Expected:")
    print(f"    Tile(2,3)[0,0]:     {expected_tile23_00}")
    print(f"    Tile(2,3)[5,5]:     {expected_tile23_55}")
    print(f"    Tile(0,0)[0,0]:     {expected_tile00_00}")
    print(f"    Tile(0,0)[31,15]:   {expected_tile00_31_15}")
    print(f"    Tile(3,3)[0,0]:     {expected_tile33_00}")
    print(f"    Tile shape:         ({TILE_M}, {TILE_N})")
    print(f"    Sum of tile[0,0]:   {expected_sum}")
    
    print(f"\n  Verification: {'✓ PASSED' if passed else '✗ FAILED'}")
    print("=" * 60 + "\n")
    
    return passed


# ─────────────────────────────────────────────
# CHECKPOINT
# ─────────────────────────────────────────────
# C1: How does local_tile differ from simple slicing like tensor[64:96, 48:64]?
# C2: In FlashAttention-2, what are the typical block sizes for Q and K/V tiles?
# C3: How would you handle a sequence length that isn't divisible by the tile size?
# C4: What is the relationship between local_tile and logical_divide?

if __name__ == "__main__":
    run()
