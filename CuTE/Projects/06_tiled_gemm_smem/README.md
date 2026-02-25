# Project 06: Tiled GEMM with Shared Memory Optimization

## Objective

Implement an optimized tiled GEMM kernel that maximizes shared memory usage for data reuse. This project teaches:
- Advanced shared memory tiling strategies
- Cooperative thread array (CTA) level optimization
- Minimizing global memory bandwidth
- Register blocking for increased arithmetic intensity

## Theory

### Why Shared Memory?

Global memory (HBM) has high latency (~400 cycles) but high bandwidth.
Shared memory has low latency (~20 cycles) and is on-chip.

By loading tiles into shared memory, we amortize the global memory cost:

```
Naive GEMM:           Tiled GEMM:
- Load A[i,k] once    - Load A tile once, reuse BK times
- Load B[k,j] once    - Load B tile once, reuse BM times
- O(N³) HBM accesses  - O(N²/B) HBM accesses
```

### Tiling Strategy

```
┌─────────────────────────────────────┐
│  Global Memory (HBM)                │
│  A[M×K], B[K×N], C[M×N]             │
└─────────────────────────────────────┘
              ↓ Load tile
┌─────────────────────────────────────┐
│  Shared Memory (SRAM)               │
│  As[BM×BK], Bs[BK×BN]               │
└─────────────────────────────────────┘
              ↓ Compute
┌─────────────────────────────────────┐
│  Registers (per thread)             │
│  C_thread[THREAD_M×THREAD_N]        │
└─────────────────────────────────────┘
```

### Optimal Tile Sizes

For RTX 4060 (SM 8.9):
- Shared memory per SM: 48 KB (configurable to 164 KB)
- Recommended: BM=128, BN=128, BK=8
- Thread block: 16×16 threads
- Each thread: 8×8 elements (64 registers max)

## Your Task

### Step 1: Shared Memory Layout

```cpp
__shared__ float As[BM * BK];
__shared__ float Bs[BK * BN];

// Use 2D indexing for clarity
#define AS(row, col) As[(row) * BK + (col)]
#define BS(row, col) Bs[(row) * BN + (col)]
```

### Step 2: Cooperative Loading

```cpp
// Each thread loads multiple elements
for (int i = tid / BN; i < BM; i += blockDim.x / BN) {
    AS(i, tid % BK) = A[global_row + i][k + tid % BK];
}
```

### Step 3: Register Blocking

```cpp
// Each thread accumulates THREAD_M × THREAD_N elements
float accum[THREAD_M][THREAD_N] = {0};

for (int kk = 0; kk < BK; kk++) {
    for (int m = 0; m < THREAD_M; m++) {
        for (int n = 0; n < THREAD_N; n++) {
            accum[m][n] += AS(local_m + m, kk) * BS(kk, local_n + n);
        }
    }
}
```

## Exercises

### Exercise 1: Basic Shared Memory Tiling (Required)

Implement tiled GEMM with:
- BM=64, BN=64, BK=8
- 16×16 thread blocks
- Each thread computes 4×4 output elements

### Exercise 2: Increased Register Blocking (Challenge)

Increase to 8×8 elements per thread:
- BM=128, BN=128, BK=8
- Requires 64 registers per thread
- Better arithmetic intensity

### Exercise 3: Shared Memory Bank Optimization (Advanced)

Add padding to avoid bank conflicts:
```cpp
__shared__ float As[BM * (BK + 1)];  // Padding
```

## Verification

```
[PASS] Tiled GEMM (Shared Memory): All elements match (max error: 0.001953)
Performance: XX GFLOPS (vs YY GFLOPS for naive)
```

## Performance Comparison

| Implementation | GFLOPS (RTX 4060) | Speedup |
|----------------|-------------------|---------|
| Naive | ~50 | 1x |
| Basic Tiled | ~200 | 4x |
| Shared Memory Opt | ~500 | 10x |
| Register Blocking | ~800 | 16x |

---

**Ready to optimize? Open `tiled_gemm_smem.cu`!**
