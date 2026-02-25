# Project 02: GEMM - General Matrix Multiply

## Objective

Implement a tiled matrix multiplication kernel (`C = A × B`) using CuTe's MMA (Matrix Multiply-Accumulate) atoms. This project teaches:
- 2D tensor layouts and indexing
- Tiled matrix multiplication
- Using Tensor Cores via CuTe MMA atoms
- Shared memory for data reuse

## Theory

### The Problem

Given matrices:
- `A` of shape `(M, K)`
- `B` of shape `(K, N)`
- Compute `C` of shape `(M, N)` where `C[i,j] = sum(A[i,k] * B[k,j])`

Naive implementation has O(N³) complexity and poor memory reuse.

### Tiled GEMM

Tiling divides the computation into blocks that fit in fast memory:

```
┌─────────────┐       ┌─────────────┐       ┌─────────────┐
│             │       │             │       │             │
│   C_tile    │   =   │   A_tile    │   ×   │   B_tile    │
│  (BM×BN)    │       │  (BM×BK)    │       │  (BK×BN)    │
│             │       │             │       │             │
└─────────────┘       └─────────────┘       └─────────────┘
```

Each thread block computes one tile of C, loading tiles of A and B into shared memory.

### CuTe MMA Atoms

CuTe provides hardware-level matrix multiply operations:

```cpp
// MMA atom: computes D = A × B + C
// A: (M,K) in shared memory, B: (K,N) in shared memory
// C, D: (M,N) in registers
cute::gemm(A, B, C, D);
```

## Your Task

### Step 1: Understand 2D Layouts

CuTe represents 2D matrices using compound layouts:

```cpp
// Row-major 2D layout: (rows, cols) with stride (cols, 1)
auto layout_2d = make_layout(make_shape(M, N), make_stride(N, 1));

// Or using CuTe's convenience function
auto layout_2d = make_layout(M, N);  // Defaults to row-major
```

### Step 2: Implement Tiled GEMM

Complete the kernel in `gemm.cu`:

```cpp
__global__ void gemm_cute(float* A, float* B, float* C, 
                          int M, int K, int N) {
    // Tile dimensions (compile-time constants)
    const int BM = 64;  // Tile rows
    const int BN = 64;  // Tile columns  
    const int BK = 8;   // Reduction dimension
    
    // TODO 1: Calculate tile indices
    int tile_m = blockIdx.y;
    int tile_n = blockIdx.x;
    
    // TODO 2: Create 2D layouts for global matrices
    // auto layout_A = make_layout(make_shape(M, K), make_stride(K, 1));
    // auto layout_B = make_layout(make_shape(K, N), make_stride(N, 1));
    // auto layout_C = make_layout(make_shape(M, N), make_stride(N, 1));
    
    // TODO 3: Create tensors from pointers
    // auto tensor_A = make_tensor(make_gmem_ptr(A), layout_A);
    // auto tensor_B = make_tensor(make_gmem_ptr(B), layout_B);
    // auto tensor_C = make_tensor(make_gmem_ptr(C), layout_C);
    
    // TODO 4: Allocate shared memory for tiles
    // __shared__ float As[BM * BK];
    // __shared__ float Bs[BK * BN];
    
    // TODO 5: Implement tiled matmul loop
    // for (int k = 0; k < K; k += BK) {
    //     // Load tiles from global to shared memory
    //     // Execute MMA operation
    //     // Accumulate results
    // }
}
```

### Step 3: Key CuTe Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `make_shape(m, n)` | Create 2D shape | `make_shape(64, 64)` |
| `make_stride(n, 1)` | Create row-major stride | `make_stride(64, 1)` |
| `make_layout(shape, stride)` | Create 2D layout | `make_layout(shape, stride)` |
| `cute::gemm()` | Matrix multiply atom | `gemm(A, B, C, D)` |

### Step 4: Implementation Hints

#### Thread Mapping for 2D Tiles

```cpp
// 2D thread block (e.g., 16x16 threads per block)
int thread_m = threadIdx.y;  // 0..15
int thread_n = threadIdx.x;  // 0..15

// Each thread computestes one element of the output tile
int out_m = tile_m * BM + thread_m;
int out_n = tile_n * BN + thread_n;
```

#### Loading Tiles

```cpp
// Load element into shared memory
int smem_idx = thread_m * BK + thread_k;
As[smem_idx] = tensor_A(row, k + thread_k);
```

## Exercises

### Exercise 1: Basic Tiled GEMM (Required)

Implement a tiled GEMM with:
- Tile size: 64×64 output
- Thread block: 16×16 threads
- Each thread computestes 4×4 elements

### Exercise 2: Shared Memory Optimization (Challenge)

Improve data reuse by:
- Loading tiles cooperatively (each thread loads one element)
- Using `__syncthreads()` to synchronize after loads
- Minimizing global memory accesses

### Exercise 3: MMA Atoms (Advanced)

Use CuTe's hardware MMA operations:

```cpp
#include <cute/algorithm/gemm.hpp>

// Define MMA atom traits
using MMA = MMA_Atom<SM80>;  // RTX 4060 uses SM80

// Execute tiled gemm using MMA
cute::gemm<MMA>(A_tile, B_tile, C_accum, D_result);
```

## Verification

Your implementation is correct if:

```
[PASS] GEMM: All elements match (max error: 0.001953)
Matrix C[0:3, 0:3]:
  2040.000  2090.000  2140.000
  2900.000  2990.000  3080.000
  3760.000  3890.000  4020.000
```

## Performance Notes

| Optimization | Speedup | Complexity |
|--------------|---------|------------|
| Naive (no tiling) | 1x | Simple |
| Tiled GEMM | 5-10x | Moderate |
| Tiled + Shared Mem | 20-50x | Complex |
| MMA Atoms | 50-100x | Expert |

## Common Pitfalls

1. **Wrong stride calculation**: Row-major stride is `(N, 1)`, not `(1, N)`
2. **Missing synchronization**: Always `__syncthreads()` after shared memory loads
3. **Index out of bounds**: Check `row < M` and `col < N` before accessing
4. **Incorrect tile indexing**: Remember `global_idx = tile_idx * tile_size + local_idx`

## Mathematical Background

Matrix multiplication computes:

```
C[i,j] = Σ A[i,k] × B[k,j]  for k = 0 to K-1
```

In tiled form, each block computes:

```
C_tile[i,j] = Σ A_tile[i,k] × B_tile[k,j]
```

## Next Steps

After completing this project:
1. Try [Project 03: Softmax](../03_softmax/) for reduction patterns
2. Experiment with different tile sizes (32×32, 128×128)
3. Explore CuTe's `tiled_gemm` algorithm in `cute/algorithm/gemm.hpp`
4. Study how CUTLASS implements production GEMM kernels

---

**Ready to multiply matrices? Open `gemm.cu` and start implementing!**
