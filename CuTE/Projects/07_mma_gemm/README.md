# Project 07: MMA GEMM with Tensor Cores

## Objective

Implement GEMM using CuTe's MMA (Matrix Multiply-Accumulate) atoms that directly utilize NVIDIA Tensor Cores. This project teaches:
- Hardware MMA instructions (WMMA)
- Tensor Core data types and layouts
- Fragment-based computation
- SM80+ Tensor Core programming

## Theory

### Tensor Cores

Tensor Cores are specialized hardware units for matrix operations:

```
// Tensor Core operation (16x16x16 for FP16)
D = A × B + C

where:
- A: 16×16 matrix
- B: 16×16 matrix
- C, D: 16×16 accumulator matrices
```

For SM 8.0+ (Ampere, Ada, Hopper):
- FP16: 16×16×16 per Tensor Core
- TF32: 16×16×8 per Tensor Core
- BF16: 16×16×16 per Tensor Core

### CuTe MMA Atoms

CuTe provides hardware abstraction for Tensor Cores:

```cpp
#include <cute/atom/mma_atom.hpp>

// Define MMA atom for SM80
using MMA = MMA_Atom<SM80>;

// Execute MMA operation
MMA::gemm(A_frag, B_frag, C_frag, D_frag);
```

### Fragment Layouts

Tensor Core operations use special fragment layouts:

```cpp
// MMA traits define the fragment layouts
using Traits = MMA_Traits<SM80>;
using Atom = MMA_Atom<Traits>;

// Fragment shapes
// A_frag: (MMA_M, MMA_K)
// B_frag: (MMA_K, MMA_N)
// C_frag, D_frag: (MMA_M, MMA_N)
```

## Your Task

### Step 1: Include MMA Headers

```cpp
#include <cute/atom/mma_atom.hpp>
#include <cute/algorithm/gemm.hpp>
```

### Step 2: Define MMA Atom

```cpp
using MMA_Atom = MMA_Atom<SM80>;
```

### Step 3: Implement MMA-based GEMM

```cpp
__global__ void mma_gemm_kernel(float* A, float* B, float* C, ...) {
    // Define MMA atom
    using MMA = MMA_Atom<SM80>;
    
    // Load data into registers (fragments)
    // Execute MMA operations
    // Store results
}
```

## Exercises

### Exercise 1: Basic MMA GEMM (Required)

Implement GEMM using CuTe's MMA atoms with:
- 16×16×16 MMA operations
- FP32 accumulation
- Proper fragment loading/storing

### Exercise 2: Tiled MMA GEMM (Challenge)

Combine MMA with tiling:
- Larger tiles (128×128)
- Multiple MMA operations per tile
- Shared memory for operands

### Exercise 3: WMMA Intrinsics (Advanced)

Use raw WMMA intrinsics:
```cpp
wmma::fragment<wmma::matrix_a, 16, 16, 16, ...> a_frag;
wmma::load_matrix_sync(a_frag, shared_A, lda);
wmma::mma_sync(d_frag, a_frag, b_frag, c_frag);
```

## Verification

```
[PASS] MMA GEMM (Tensor Cores): All elements match (max error: 0.003906)
Note: Tensor Core operations have slightly higher numerical error
```

## Performance Comparison

| Implementation | GFLOPS (RTX 4060) | Speedup |
|----------------|-------------------|---------|
| Naive FP32 | ~50 | 1x |
| Shared Memory FP32 | ~500 | 10x |
| MMA Tensor Cores | ~2000+ | 40x |

---

**Ready to accelerate with Tensor Cores? Open `mma_gemm.cu`!**
