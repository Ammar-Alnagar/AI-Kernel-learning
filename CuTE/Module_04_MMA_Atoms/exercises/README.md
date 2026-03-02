# Module 04: MMA Atoms - Exercises

## Overview
This directory contains hands-on exercises to practice CuTe MMA (Matrix Multiply-Accumulate) Atom concepts. MMA atoms are the fundamental building blocks for Tensor Core operations.

## Building the Exercises

### Prerequisites
- CUDA Toolkit with sm_89 support
- CUTLASS library with CuTe headers
- Complete Module 01-03 exercises

### Build Instructions

```bash
cd exercises
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Exercises

```bash
./ex01_mma_atom_basics
./ex02_tensor_core_sim
```

---

## Exercises

### Exercise 01: MMA Atom Basics
**File:** `ex01_mma_atom_basics.cu`

**Learning Objectives:**
- Understand MMA fundamentals (D = A × B + C)
- Learn Tensor Core concepts
- Practice small matrix multiplication
- See warp-level cooperation

**Step-by-Step Guidance:**
1. **Understand the operation** - D = A × B + C
2. **Create operand matrices** - A (M×K), B (K×N), C (M×N)
3. **Perform multiplication** - Manual or Tensor Core
4. **Verify results** - Check D against expected

**Key Concepts:**
- **MMA:** Matrix Multiply-Accumulate
- **Tensor Cores:** Hardware units for fast matrix math
- **Warp-Level:** 32 threads cooperate

**MMA Operation:**
```
D[i,j] = C[i,j] + Σ(k=0 to K-1) A[i,k] × B[k,j]

For 2×4 × 4×2 = 2×2:
D[0,0] = C[0,0] + A[0,0]*B[0,0] + A[0,1]*B[1,0] + A[0,2]*B[2,0] + A[0,3]*B[3,0]
```

**Common Pattern:**
```cpp
// Manual 2×2 matrix multiply
for (int i = 0; i < 2; ++i)
    for (int j = 0; j < 2; ++j)
        for (int k = 0; k < 4; ++k)
            D[i][j] += A[i][k] * B[k][j];
```

**Common Pitfalls:**
- Dimension mismatch (K must match)
- Not initializing accumulator to zero
- Integer overflow in accumulation

**Expected Output:**
Matrix multiplication showing operands and result.

**Verification:**
- Result dimensions: (M×N)
- Each element: dot product of row and column

**Extension Challenge:**
Try 4×4 × 4×4 multiplication.

**Concepts:** MMA operation, Tensor Cores, warp-level

**Estimated Time:** 20-25 minutes

---

### Exercise 02: Tensor Core Operation Simulation
**File:** `ex02_tensor_core_sim.cu`

**Learning Objectives:**
- Simulate Tensor Core operations
- Compare throughput with CUDA cores
- Understand performance benefits
- Count operations for efficiency

**Step-by-Step Guidance:**
1. **Understand Tensor Cores** - Specialized matrix units
2. **Simulate operation** - Software emulation
3. **Compare throughput** - Tensor vs CUDA cores
4. **Calculate speedup** - Theoretical vs actual

**Key Concepts:**
- **Tensor Core Throughput:** Much higher than CUDA cores
- **Operation Counting:** FLOPs per cycle
- **Mixed Precision:** FP16 compute, FP32 accumulate

**Throughput Comparison (A100):**
```
FP32 CUDA Cores:  19.5 TFLOPS
FP16 Tensor Cores: 312 TFLOPS (16× faster!)
```

**Common Pitfalls:**
- Expecting exact FP32 accuracy from FP16
- Not accounting for data conversion overhead
- Ignoring memory bandwidth limits

**Expected Output:**
Throughput comparison and operation counting.

**Verification:**
- Tensor Core simulation should match manual MMA
- Speedup should reflect hardware capabilities

**Extension Challenge:**
Calculate theoretical peak TFLOPS for your GPU.

**Concepts:** Simulation, throughput, acceleration

**Estimated Time:** 20-25 minutes

---

### Exercise 03: Thread to Tensor Core Mapping
**File:** `ex03_thread_tensor_mapping.cu`

**Learning Objectives:**
- Understand warp-level organization
- Learn thread roles in MMA
- See operand loading patterns
- Map threads to tensor operations

**Step-by-Step Guidance:**
1. **Understand warp structure** - 32 threads
2. **Assign thread roles** - Who loads what
3. **Map to Tensor Core** - Thread → MMA mapping
4. **Coordinate loading** - Cooperative operand load

**Key Concepts:**
- **Warp:** 32 threads executing together
- **Thread Roles:** Different threads load different operands
- **Cooperation:** All threads contribute to MMA

**Thread Organization for 16×16×16 MMA:**
```
Warp (32 threads):
- 16 threads load A operands
- 16 threads load B operands
- All 32 threads participate in MMA
- Results distributed across warp
```

**Common Pattern:**
```cpp
// Thread to operand mapping
int lane_id = threadIdx.x % 32;
if (lane_id < 16) {
    // Load A operand
    load_A_fragment(a_frag[lane_id]);
} else {
    // Load B operand
    load_B_fragment(b_frag[lane_id - 16]);
}
```

**Common Pitfalls:**
- Incorrect lane ID calculation
- Not synchronizing before MMA
- Uneven work distribution

**Expected Output:**
Thread-to-Tensor-Core mapping visualization.

**Verification:**
- Each thread should have assigned role
- All operands should be loaded

**Extension Challenge:**
Map multiple warps to larger MMA operations.

**Concepts:** Warp, thread mapping, cooperation

**Estimated Time:** 25-30 minutes

---

### Exercise 04: Accumulator Management
**File:** `ex04_accumulator_management.cu`

**Learning Objectives:**
- Manage accumulator registers
- Implement multi-step accumulation
- Handle precision correctly
- Optimize register usage

**Step-by-Step Guidance:**
1. **Allocate accumulators** - Register allocation
2. **Initialize accumulators** - Zero or bias
3. **Accumulate over K** - Multiple MMA steps
4. **Store results** - Write to memory

**Key Concepts:**
- **Accumulator:** Holds running sum (FP32)
- **Register Allocation:** Limited resource
- **K-Reduction:** Sum over K dimension

**Accumulation Pattern:**
```cpp
// Initialize accumulator to zero
float accum[4] = {0, 0, 0, 0};

// Multi-step accumulation
for (int k_tile = 0; k_tile < K / 16; ++k_tile) {
    load_operands(a_frag, b_frag, k_tile);
    mma_sync(accum, a_frag, b_frag);  // accum += a × b
}

// Store final result
store_results(accum, C);
```

**Register Budget:**
```
For 16×16×16 MMA:
- Accumulator: 256 bits (8 × FP32)
- Operand A: 128 bits
- Operand B: 128 bits
- Total per thread: ~16 registers
```

**Common Pitfalls:**
- Not initializing accumulators
- Register spilling (too many registers)
- Precision loss in accumulation

**Expected Output:**
Accumulator management and multi-step MMA.

**Verification:**
- Accumulator should start at zero
- Final result should match full matrix multiply

**Extension Challenge:**
Implement accumulator with bias addition.

**Concepts:** Accumulators, registers, K-reduction

**Estimated Time:** 25-30 minutes

---

### Exercise 05: Mixed Precision MMA
**File:** `ex05_mixed_precision_mma.cu`

**Learning Objectives:**
- Use mixed precision (FP16/FP32)
- Understand BF16, INT8 configurations
- Select appropriate precision
- Handle type conversions

**Step-by-Step Guidance:**
1. **Create FP16 operands** - Half precision inputs
2. **Use FP32 accumulator** - Full precision output
3. **Perform mixed MMA** - FP16 × FP16 → FP32
4. **Compare precisions** - FP16 vs BF16 vs INT8

**Key Concepts:**
- **Mixed Precision:** Different types for different parts
- **FP16:** 16-bit float (fast, less range)
- **BF16:** Brain float (more range, less precision)
- **INT8:** 8-bit integer (inference)

**Precision Comparison:**
| Type | Bits | Range | Precision | Use Case |
|------|------|-------|-----------|----------|
| FP32 | 32 | Wide | High | Accumulation |
| FP16 | 16 | Medium | Medium | Training |
| BF16 | 16 | Wide | Low | ML Training |
| INT8 | 8 | Low | Low | Inference |

**Common Pattern:**
```cpp
// Mixed precision MMA
half a_frag[8];   // FP16 operand A
half b_frag[8];   // FP16 operand B
float accum[4];   // FP32 accumulator

// MMA: FP16 × FP16 → FP32
mma_sync(accum, a_frag, b_frag);
```

**Common Pitfalls:**
- Overflow in FP16 (limited range)
- Precision loss for small values
- Not handling type conversions

**Expected Output:**
Mixed precision MMA with comparison.

**Verification:**
- Results should be reasonable (check for overflow)
- FP32 accumulation should preserve precision

**Extension Challenge:**
Implement INT8 MMA for inference.

**Concepts:** Mixed precision, FP16, BF16, INT8

**Estimated Time:** 25-30 minutes

---

### Exercise 06: GEMM with MMA Atoms
**File:** `ex06_gemm_with_mma.cu`

**Learning Objectives:**
- Build complete GEMM kernel
- Implement multi-level tiling
- Handle K-dimension reduction
- Structure full GEMM

**Step-by-Step Guidance:**
1. **Tile the problem** - Break into MMA-sized chunks
2. **Load operands** - From global to shared memory
3. **Execute MMA** - Tensor Core operations
4. **Accumulate results** - Over K dimension

**Key Concepts:**
- **GEMM:** GEneral Matrix Multiply (C = A × B)
- **Multi-level Tiling:** Block → Warp → MMA
- **K-Reduction:** Sum over shared dimension

**GEMM Structure:**
```
Full GEMM (M×K × K×N = M×N):
1. Divide into blocks (tile_M × tile_N)
2. Each block loads tile to shared memory
3. Warp performs MMA on sub-tiles
4. Accumulate over K dimension
```

**Complete GEMM Pattern:**
```cpp
__global__ void gemm(float* A, float* B, float* C, int M, int K, int N) {
    // Shared memory for tiles
    extern __shared__ float smem[];
    float* As = smem;
    float* Bs = &smem[TILE_M * TILE_K];
    
    // Accumulator
    float accum[MMA_M * MMA_N] = {0};
    
    // Mainloop over K
    for (int k = 0; k < K; k += TILE_K) {
        // Load tiles
        load_tile(A, As, k);
        load_tile(B, Bs, k);
        __syncthreads();
        
        // MMA operation
        mma(accum, As, Bs);
    }
    
    // Store result
    store(C, accum);
}
```

**Common Pitfalls:**
- Incorrect tile sizing
- Not handling remainder K
- Shared memory bank conflicts

**Expected Output:**
Complete GEMM implementation with results.

**Verification:**
- Result should match reference GEMM
- Performance should improve with larger matrices

**Extension Challenge:**
Add support for non-multiple-of-tile dimensions.

**Concepts:** GEMM, tiling, reduction

**Estimated Time:** 35-40 minutes

---

## Learning Path

1. **Exercise 01** - MMA basics
2. **Exercise 02** - Tensor Core simulation
3. **Exercise 03** - Thread mapping
4. **Exercise 04** - Accumulator management
5. **Exercise 05** - Mixed precision
6. **Exercise 06** - Complete GEMM

## MMA Configurations Summary

### Common sm_80 Configurations

| Configuration | M×N×K | Types | Use Case |
|---------------|-------|-------|----------|
| SM80_16x8x16 | 16×8×16 | F32/F16/F16/F32 | General GEMM |
| SM80_16x8x32 | 16×8×32 | F32/F16/F16/F32 | K-parallel |
| SM80_8x8x4 | 8×8×4 | F32/FP64/FP64/FP64 | Scientific |
| SM80_16x8x32 | 16×8×32 | S32/S8/S8/S32 | INT8 Inference |

### Architecture Support

| Arch | GPU | FP16 | INT8 | FP64 |
|------|-----|------|------|------|
| sm_70 | V100 | ✓ | ✓ | ✗ |
| sm_75 | T4 | ✓ | ✓ | ✗ |
| sm_80 | A100 | ✓ | ✓ | ✓ |
| sm_86 | A10 | ✓ | ✓ | ✓ |
| sm_89 | H100 | ✓ | ✓ | ✓ |

## Key Formulas

### GEMM Complexity
```
Total FLOPs = 2 × M × N × K
MMA Operations = (M/16) × (N/16) × (K/16)  [for 16×16×16 MMA]
```

### Throughput Calculation
```
Peak TFLOPS = Ops/clock × Warps/SM × SMs × Frequency
A100 FP16 = 512 × 8 × 108 × 1.4 GHz ≈ 312 TFLOPS
```

## Tips for Success

1. **Understand warp organization** - 32 threads cooperate
2. **Manage registers carefully** - Limited resource
3. **Use mixed precision** - FP16 inputs, FP32 accumulation
4. **Tile appropriately** - Match MMA atom size
5. **Profile configurations** - Different configs for different sizes

## Next Steps

After completing these exercises:
1. Move to Module 05: Shared Memory Swizzling
2. Learn bank conflict avoidance
3. Study swizzling techniques

## Additional Resources

- Module 04 README.md - Concept overview
- `mma_atom_basics.cu` - Reference implementation
- CUTLASS documentation - https://github.com/NVIDIA/cutlass
- Tensor Core Programming Guide - NVIDIA Developer
