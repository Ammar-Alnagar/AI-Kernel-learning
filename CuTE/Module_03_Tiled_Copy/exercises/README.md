# Module 03: Tiled Copy - Exercises

## Overview
This directory contains hands-on exercises to practice CuTe Tiled Copy concepts. Tiled copy enables efficient data movement through thread cooperation and vectorized operations.

## Building the Exercises

### Prerequisites
- CUDA Toolkit with sm_89 support
- CUTLASS library with CuTe headers
- Complete Module 01-02 exercises

### Build Instructions

```bash
cd exercises
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Exercises

```bash
./ex01_tiled_copy_basics
./ex02_vectorized_loads
```

---

## Exercises

### Exercise 01: Tiled Copy Basics
**File:** `ex01_tiled_copy_basics.cu`

**Learning Objectives:**
- Understand what tiled copy means
- See how threads divide copy work
- Practice with simple tile copy patterns
- Calculate copy efficiency

**Step-by-Step Guidance:**
1. **Understand the concept** - Copy data in tiles, not element-by-element
2. **Visualize tile division** - See how 8×8 matrix divides into 4×4 tiles
3. **Simulate tiled copy** - Copy one tile at a time
4. **Calculate efficiency** - Compare with naive copy

**Key Concepts:**
- **Tiled Copy:** Copying data in rectangular tiles
- **Thread Cooperation:** Multiple threads work together
- **Efficiency:** Better cache utilization, vectorized loads

**Visual Example:**
```
8×8 Matrix divided into 2×2 tiles (each 4×4):
┌────────┬────────┐
│ Tile 00│ Tile 01│
│ (4×4)  │ (4×4)  │
├────────┼────────┤
│ Tile 10│ Tile 11│
│ (4×4)  │ (4×4)  │
└────────┴────────┘
```

**Common Pattern:**
```cpp
// Each thread copies one tile
for (int i = 0; i < TILE_SIZE; ++i)
    for (int j = 0; j < TILE_SIZE; ++j)
        dst[i][j] = src[i][j];
```

**Common Pitfalls:**
- Tile size not matching thread block size
- Not handling boundary cases
- Ignoring alignment requirements

**Expected Output:**
Tile division visualization and copy simulation.

**Verification:**
- All tiles should be copied correctly
- Destination should match source

**Extension Challenge:**
Try different tile sizes (2×2, 8×8) and compare.

**Concepts:** Tiling, work distribution, parallel copy

**Estimated Time:** 15-20 minutes

---

### Exercise 02: Vectorized Loads
**File:** `ex02_vectorized_loads.cu`

**Learning Objectives:**
- Use vectorized memory operations (float4)
- Understand alignment requirements
- Measure bandwidth improvement
- Apply vectorization in copy kernels

**Step-by-Step Guidance:**
1. **Understand vectorization** - Load 4 floats in one instruction
2. **Check alignment** - Data must be 16-byte aligned
3. **Implement vectorized load** - Use `float4` type
4. **Compare bandwidth** - Measure improvement

**Key Concepts:**
- **Vectorized Load:** 128-bit load (4 × 32-bit floats)
- **Alignment:** Address must be multiple of 16 bytes
- **Bandwidth:** 4× improvement over scalar loads

**Visual Example:**
```
Scalar loads (4 instructions):
LD.F32 r1, [addr]
LD.F32 r2, [addr+4]
LD.F32 r3, [addr+8]
LD.F32 r4, [addr+12]

Vectorized load (1 instruction):
LD.F32.F4 r1, [addr]  // Loads 16 bytes at once
```

**Common Pattern:**
```cpp
// Vectorized load
float4 val = reinterpret_cast<float4*>(&src[idx])[0];
reinterpret_cast<float4*>(&dst[idx])[0] = val;

// Check alignment
bool aligned = (uintptr_t(ptr) % 16 == 0);
```

**Common Pitfalls:**
- Unaligned addresses cause crashes
- Not checking alignment before vectorized load
- Mixing scalar and vectorized access

**Expected Output:**
Comparison of scalar vs vectorized load performance.

**Verification:**
- Vectorized loads should be 4× faster
- Data should be identical after copy

**Extension Challenge:**
Implement fallback to scalar loads for unaligned data.

**Concepts:** Vectorization, alignment, bandwidth

**Estimated Time:** 20-25 minutes

---

### Exercise 03: Thread Cooperation
**File:** `ex03_thread_cooperation.cu`

**Learning Objectives:**
- Understand thread collaboration patterns
- Learn thread indexing in blocks
- Configure blocks for tiled copy
- Divide work among threads

**Step-by-Step Guidance:**
1. **Understand thread indices** - threadIdx, blockIdx
2. **Calculate global position** - Map thread to data
3. **Divide work** - Each thread copies portion
4. **Verify cooperation** - All threads work together

**Key Concepts:**
- **Thread Index:** Unique ID within block
- **Block Index:** Unique ID within grid
- **Work Division:** Fair distribution among threads

**Thread Indexing Formula:**
```cpp
int global_row = blockIdx.y * blockDim.y + threadIdx.y;
int global_col = blockIdx.x * blockDim.x + threadIdx.x;
```

**Common Pattern:**
```cpp
__global__ void tiled_copy(float* src, float* dst, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        dst[row * N + col] = src[row * N + col];
    }
}
```

**Common Pitfalls:**
- Out-of-bounds access without checking
- Incorrect thread-to-data mapping
- Load imbalance among threads

**Expected Output:**
Thread cooperation visualization and work division.

**Verification:**
- Each thread should copy unique data
- All data should be copied exactly once

**Extension Challenge:**
Implement 2D thread block for 2D data.

**Concepts:** Cooperation, indexing, parallelism

**Estimated Time:** 20-25 minutes

---

### Exercise 04: Global to Shared Memory Copy
**File:** `ex04_gmem_to_smem.cu`

**Learning Objectives:**
- Master gmem → smem transfers
- Implement tiled loading patterns
- Ensure coalesced access
- Use shared memory effectively

**Step-by-Step Guidance:**
1. **Allocate shared memory** - `extern __shared__`
2. **Load from global** - Coalesced pattern
3. **Store to shared** - Bank-conflict-free
4. **Synchronize** - `__syncthreads()`

**Key Concepts:**
- **Global Memory:** Large, slow, off-chip
- **Shared Memory:** Small, fast, on-chip
- **Coalesced Load:** Consecutive threads, consecutive addresses

**Common Pattern:**
```cpp
__global__ void load_to_smem(float* gmem, float* smem, int M, int N) {
    extern __shared__ float smem_buffer[];
    
    // Coalesced load from global
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        smem_buffer[row * N + col] = gmem[row * N + col];
    }
    
    __syncthreads();  // Wait for all threads
}
```

**Common Pitfalls:**
- Forgetting `__syncthreads()`
- Bank conflicts in shared memory
- Exceeding shared memory limits

**Expected Output:**
Gmem to smem transfer with timing comparison.

**Verification:**
- Shared memory should contain correct data
- All threads should see consistent data after sync

**Extension Challenge:**
Implement double buffering with two shared memory buffers.

**Concepts:** Memory hierarchy, tiling, coalescing

**Estimated Time:** 25-30 minutes

---

### Exercise 05: Copy Atom and Tiled Copy
**File:** `ex05_copy_atom.cu`

**Learning Objectives:**
- Understand CuTe's copy atom abstraction
- Learn thread organization for atoms
- See instruction selection
- Use atoms in tiled copy

**Key Concepts:**
- **Copy Atom:** Smallest unit of copy operation
- **Abstraction:** Portable across architectures
- **Instruction Selection:** Hardware-specific instructions

**What is a Copy Atom:**
```
Copy Atom = (Thread Layout, Data Layout, Instruction Type)

It defines:
- How threads are organized
- How data is laid out
- Which hardware instruction to use
```

**Common Pattern:**
```cpp
// Conceptual copy atom usage
auto atom = make_copy_atom(thread_layout, data_layout);
copy(atom, src_tensor, dst_tensor);
```

**Common Pitfalls:**
- Mismatched thread and data layouts
- Not understanding atom configuration
- Using wrong atom for data type

**Expected Output:**
Copy atom structure and usage examples.

**Verification:**
- Atom should correctly copy data
- Thread organization should match layout

**Extension Challenge:**
Configure atom for different data types (float, half, int8).

**Concepts:** Atoms, abstraction, portability

**Estimated Time:** 25-30 minutes

---

### Exercise 06: Coalescing Strategies
**File:** `ex06_coalescing_strategies.cu`

**Learning Objectives:**
- Optimize memory access patterns
- Compare coalesced vs uncoalesced access
- Understand layout impact on coalescing
- Apply best practices

**Step-by-Step Guidance:**
1. **Create row-major layout** - Test row access
2. **Create column-major layout** - Test column access
3. **Measure coalescing** - Compare efficiency
4. **Apply best practices** - Match layout to access

**Key Concepts:**
- **Coalesced:** Consecutive threads → consecutive addresses
- **Uncoalesced:** Scattered access pattern
- **Efficiency:** Coalesced = maximum bandwidth

**Access Pattern Analysis:**
```
Row-Major Layout (4×4):
Row access:    0 1 2 3  ✓ Coalesced
Column access: 0 4 8 12 ✗ Uncoalesced

Column-Major Layout (4×4):
Row access:    0 4 8 12 ✗ Uncoalesced
Column access: 0 1 2 3  ✓ Coalesced
```

**Common Pitfalls:**
- Mismatched layout and access pattern
- Not considering all access patterns
- Ignoring performance impact

**Expected Output:**
Coalescing analysis for different patterns.

**Verification:**
- Coalesced access should show stride-1 pattern
- Uncoalesced should show larger strides

**Extension Challenge:**
Design layout for mixed access patterns.

**Concepts:** Coalescing, optimization, bandwidth

**Estimated Time:** 25-30 minutes

---

## Learning Path

1. **Exercise 01** - Tiled copy basics
2. **Exercise 02** - Vectorized loads
3. **Exercise 03** - Thread cooperation
4. **Exercise 04** - gmem to smem
5. **Exercise 05** - Copy atoms
6. **Exercise 06** - Coalescing strategies

## Key Concepts Summary

### Tiled Copy Benefits
- Better memory coalescing
- Enables vectorized loads
- Thread cooperation
- Overlap with computation

### Memory Access Patterns
| Pattern | Row-Major | Column-Major |
|---------|-----------|--------------|
| Row Access | Coalesced | Uncoalesced |
| Column Access | Uncoalesced | Coalesced |

### Vectorized Load Requirements
- Address must be 16-byte aligned
- Data type must be compatible (float4, int4)
- All 4 elements should be needed

## Tips for Success

1. **Match layout to access pattern** for coalescing
2. **Use vectorized loads** when possible (4× bandwidth)
3. **Tile your data** for shared memory reuse
4. **Overlap copy with compute** using async operations
5. **Always synchronize** after shared memory writes

## Common Patterns Reference

```cpp
// Basic tiled copy pattern
__global__ void tiled_copy(float* src, float* dst, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        dst[row * N + col] = src[row * N + col];
    }
}

// Vectorized load pattern
float4 val = reinterpret_cast<float4*>(&src[idx])[0];
reinterpret_cast<float4*>(&dst[idx])[0] = val;

// Shared memory load with sync
extern __shared__ float smem[];
smem[threadIdx.y * N + threadIdx.x] = gmem[...];
__syncthreads();
```

## Next Steps

After completing these exercises:
1. Move to Module 04: MMA Atoms
2. Learn Tensor Core operations
3. Study matrix multiplication

## Additional Resources

- Module 03 README.md - Concept overview
- `tiled_copy_basics.cu` - Reference implementation
- CuTe documentation - https://github.com/NVIDIA/cutlass
- CUDA Programming Guide - Memory Transactions chapter
