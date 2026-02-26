#include "cute/layout.hpp"
#include "cute/util/print.hpp"
#include <iostream>

using namespace cute;

int main() {
  std::cout << "=== Exercise 06: Hierarchical Layouts ===" << std::endl;

  // =========================================================================
  // TASK 1: Thread Block Layout
  // =========================================================================
  // 4 warps x 32 lanes, row-major means warp_id is the slow dimension
  // stride (32, 1): moving to next warp costs 32 offsets, next lane costs 1
  auto block_layout = make_layout(make_shape(Int<4>{}, Int<32>{}),
                                  make_stride(Int<32>{}, Int<1>{}));

  std::cout << "\n--- Task 1: Block Layout ---" << std::endl;
  std::cout << "block_layout: ";
  print(block_layout);
  std::cout << std::endl;
  // Manual: offset(warp=2, lane=5) = 2*32 + 5*1 = 69
  std::cout << "Manual offset (warp=2, lane=5): 2*32 + 5*1 = 69" << std::endl;
  std::cout << "CuTe offset   (warp=2, lane=5): " << block_layout(2, 5)
            << std::endl;
  // Both must print 69. If they don't, your stride is wrong.

  // =========================================================================
  // TASK 2: 2D Thread Layout
  // =========================================================================
  // (8 rows, 16 cols) of threads, row-major → stride (16, 1)
  // Why (8,16) over (16,8)?
  //   In row-major data, threads in the same row should span consecutive cols
  //   so that a warp's 32 lanes read 32 consecutive floats → coalesced.
  //   (8,16): 16 consecutive threads per row → one warp spans 2 rows of 16.
  //   (16,8): only 8 consecutive threads per row → harder to coalesce a full
  //   warp.
  auto block_2d = make_layout(make_shape(Int<8>{}, Int<16>{}),
                              make_stride(Int<16>{}, Int<1>{}));

  std::cout << "\n--- Task 2: 2D Block Layout ---" << std::endl;
  std::cout << "block_2d: ";
  print(block_2d);
  std::cout << std::endl;
  // Manual: offset(3, 7) = 3*16 + 7*1 = 55
  std::cout << "Manual offset (3,7): 3*16 + 7*1 = 55" << std::endl;
  std::cout << "CuTe offset   (3,7): " << block_2d(3, 7) << std::endl;

  // =========================================================================
  // TASK 3: Warp/Lane Decomposition
  // =========================================================================
  // For row-major block_layout with stride (32,1):
  //   block_layout(warp_id, lane_id) = warp_id*32 + lane_id = thread_id
  // So the offset IS the thread_id. This is the definition of row-major here.
  //
  // If it were col-major (stride (1,4)):
  //   block_layout(warp_id, lane_id) = warp_id*1 + lane_id*4
  //   → offset != thread_id → warp lanes would NOT access contiguous memory

  std::cout << "\n--- Task 3: Warp/Lane Decomposition ---" << std::endl;
  bool all_match = true;
  for (int tid = 0; tid < 128; ++tid) {
    int warp_id = tid / 32; // which warp  (0..3)
    int lane_id = tid % 32; // which lane within warp (0..31)
    int offset = block_layout(warp_id, lane_id);
    if (offset != tid) {
      std::cout << "MISMATCH at tid=" << tid << " warp=" << warp_id
                << " lane=" << lane_id << " offset=" << offset << std::endl;
      all_match = false;
    }
  }
  if (all_match)
    std::cout << "All 128 threads verified: block_layout(warp,lane) == tid ✓"
              << std::endl;
  // KEY INSIGHT: row-major stride means flat thread_id == memory offset.
  // This is exactly what you want for coalesced gmem reads:
  // consecutive tids → consecutive offsets → one cache line transaction per
  // warp.

  // =========================================================================
  // TASK 4: 3-Level Hierarchy — Grid -> Block -> Warp
  // =========================================================================
  //
  // Output matrix: 64 x 64
  // Block tile:    16 x 32  →  grid is (64/16) x (64/32) = 4 x 2 blocks
  // Warp tile:      8 x 16  →  block has (16/8) x (32/16) = 2 x 2 warps
  // Lane tile:      1 x  1  →  warp has 8x16=128 lanes...
  //
  // WAIT — pause here. 8*16 = 128 lanes per warp, but a warp has 32 lanes.
  // This setup is geometrically wrong for a real kernel.
  // We keep it to practice the layout math, but flag it:
  // A correct warp tile for 32 lanes would be e.g. (4,8) or (2,16) or (1,32).
  // We'll use (8,4) → 32 lanes as the corrected lane layout below.
  //
  // For the hierarchy exercise we keep the GIVEN numbers and just do the math:

  // Grid layout: 4 blocks in row dim, 2 blocks in col dim
  // Strides in units of BLOCKS (logical, not elements):
  //   moving to next block-row costs 1 block-row unit
  //   moving to next block-col costs 1 block-col unit
  // (We track block indices here, not element offsets)
  auto grid_layout = make_layout(make_shape(Int<4>{}, Int<2>{}),
                                 make_stride(Int<2>{}, Int<1>{}));
  // Row-major over blocks: block(1,0) → index 2, block(0,1) → index 1

  // Warp layout within a block: 2 warps in row dim, 2 warps in col dim
  auto warp_layout = make_layout(make_shape(Int<2>{}, Int<2>{}),
                                 make_stride(Int<2>{}, Int<1>{}));

  // Lane layout within a warp: 8 x 4 = 32 lanes (corrected from 8x16)
  // stride (4,1): next lane-row costs 4, next lane-col costs 1
  auto lane_layout = make_layout(make_shape(Int<8>{}, Int<4>{}),
                                 make_stride(Int<4>{}, Int<1>{}));

  std::cout << "\n--- Task 4: 3-Level Hierarchy ---" << std::endl;
  std::cout << "grid_layout : ";
  print(grid_layout);
  std::cout << std::endl;
  std::cout << "warp_layout : ";
  print(warp_layout);
  std::cout << std::endl;
  std::cout << "lane_layout : ";
  print(lane_layout);
  std::cout << std::endl;

  // Manual global element calculation for block(1,1), warp(0,1), lane(5,3):
  //   Global row = block_row * 16 + warp_row * 8 + lane_row * 1
  //              = 1*16       + 0*8          + 5*1         = 21
  //   Global col = block_col * 32 + warp_col * 16 + lane_col * 1
  //              = 1*32       + 1*16          + 3*1         = 51
  //   → This thread owns output element (21, 51)
  std::cout << "block(1,1)+warp(0,1)+lane(5,3) → output element (21, 51)"
            << std::endl;

  // =========================================================================
  // TASK 5: Layout Composition — The CuTe Way
  // =========================================================================
  // Instead of 3 separate index calculations, compose into ONE layout.
  // The shape encodes the hierarchy: ((warp_dim, lane_dim), (warp_dim,
  // lane_dim))
  //
  // For the ROW dimension:
  //   warp steps by 8 elements (warp tile height)
  //   lane steps by 1 element  (lane tile height = 1)
  //   → warp row stride in output matrix (row-major, 64 cols wide) = 8 * 64 =
  //   512? NO — stride is in ELEMENTS of the output matrix, not tiles. warp_row
  //   stride = warp_tile_rows * output_cols = 8 * 64 = 512  ← if we were
  //   striding in flat memory. But here we think in 2D output coords: warp row
  //   stride = 8  (each warp-row step moves 8 rows in output) lane row stride =
  //   1  (each lane-row step moves 1 row in output)
  //
  // For the COL dimension:
  //   warp col stride = 16 (each warp-col step moves 16 cols in output)
  //   lane col stride = 1  (each lane-col step moves 1 col in output)
  //
  // Shape:  ((warp_rows=2, lane_rows=8), (warp_cols=2, lane_cols=4))
  // Stride: ((warp_row_stride=8, lane_row_stride=1), (warp_col_stride=16,
  // lane_col_stride=1))

  auto composed = make_layout(
      make_shape(make_shape(Int<2>{}, Int<8>{}),  // row: 2 warps, 8 lanes each
                 make_shape(Int<2>{}, Int<4>{})), // col: 2 warps, 4 lanes each
      make_stride(
          make_stride(Int<8>{}, Int<1>{}),  // row strides: warp=8, lane=1
          make_stride(Int<16>{}, Int<1>{})) // col strides: warp=16, lane=1
  );

  std::cout << "\n--- Task 5: Composed Layout ---" << std::endl;
  std::cout << "composed: ";
  print(composed);
  std::cout << std::endl;

  // Call with hierarchical coordinate: warp=(0,1), lane=(5,3)
  // Expected: row = 0*8 + 5*1 = 5,  col = 1*16 + 3*1 = 19
  // This is the INTRA-BLOCK offset, not global. To get global add block offset.
  auto coord = make_coord(make_coord(0, 5),  // row: warp_row=0, lane_row=5
                          make_coord(1, 3)); // col: warp_col=1, lane_col=3
  std::cout << "composed(warp_row=0,lane_row=5 | warp_col=1,lane_col=3): "
            << composed(coord) << std::endl;
  // This prints a 2D coord (5, 19) — the element this thread owns within the
  // block tile Add block(1,1) offset: global = (1*16 + 5, 1*32 + 19) = (21, 51)
  // ✓ matches Task 4

  // =========================================================================
  // SYNTHESIS ANSWERS
  // =========================================================================
  //
  // Q1: Col-major block_layout would have stride (1, 4) for shape (4,32).
  //     block_layout(warp_id, lane_id) = warp_id*1 + lane_id*4 ≠ tid
  //     Consecutive tids would map to non-consecutive offsets → uncoalesced.
  //     Row-major wins for gmem coalescing: consecutive lanes = consecutive
  //     addresses.
  //
  // Q2: Warp tile (8,16) = 128 elements ≠ 32 lanes. Fixed options:
  //     (1,32): all 32 lanes in one row → best col coalescing, poor row reuse
  //     (2,16): 2 rows of 16 → balanced, common in practice
  //     (4,8) : 4 rows of 8  → more row reuse, less col coalescing
  //     (8,4) : used above   → max row reuse, minimal col coalescing
  //     Choice depends on whether your kernel is bandwidth or compute bound.
  //
  // Q3: sm_89 MMA atom shape is (16,8,16) → output tile per warp is (16,8).
  //     Your warp layout must decompose into 16 rows x 8 cols = 128... still
  //     not 32. Reality: MMA operates on REGISTER tiles, not one element per
  //     lane. Each lane holds multiple accumulators. This is Module 04
  //     territory. The key bridge: your composed layout's lane dimension will
  //     be replaced by a register-level layout inside the MMA atom abstraction.

  std::cout << "\n=== Exercise Complete ===" << std::endl;
  return 0;
}
