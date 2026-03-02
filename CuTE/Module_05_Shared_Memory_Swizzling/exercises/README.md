# Module 05: Shared Memory Swizzling - Exercises

## Overview
This directory contains hands-on exercises to practice Shared Memory and Swizzling concepts. Learn to avoid bank conflicts and optimize shared memory access patterns.

## Building the Exercises

### Prerequisites
- CUDA Toolkit with sm_89 support
- CUTLASS library with CuTe headers
- Complete Module 01-04 exercises

### Build Instructions

```bash
cd exercises
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Exercises

```bash
./ex01_shared_memory_basics
./ex02_bank_conflict_analysis
```

---

## Exercises

### Exercise 01: Shared Memory Basics
**File:** `ex01_shared_memory_basics.cu`

**Learning Objectives:**
- Understand shared memory characteristics
- Learn bank structure (32 banks)
- Identify common use cases
- Compare latency with global memory

**Step-by-Step Guidance:**
1. **Allocate shared memory** - `extern __shared__`
2. **Understand banks** - 32 parallel memory units
3. **Load and store** - Basic operations
4. **Measure latency** - Compare with global memory

**Key Concepts:**
- **Shared Memory:** On-chip, low latency, block-scoped
- **Banks:** 32 parallel memory units (4 bytes each)
- **Bank Conflict:** Multiple threads access same bank

**Memory Hierarchy:**
```
Register (fastest, smallest)
    ↓
Shared Memory (fast, small, shared by block)
    ↓
L1/L2 Cache
    ↓
Global Memory (slowest, largest)
```

**Common Pattern:**
```cpp
__global__ void shared_example() {
    extern __shared__ float smem[];
    
    // Load from global to shared
    smem[threadIdx.x] = gmem[threadIdx.x];
    __syncthreads();
    
    // Use shared memory
    float val = smem[threadIdx.x];
}
```

**Common Pitfalls:**
- Forgetting `__syncthreads()`
- Exceeding shared memory limits
- Not considering bank conflicts

**Expected Output:**
Shared memory usage and latency comparison.

**Verification:**
- Data should be correctly loaded/stored
- Shared memory should be faster than global

**Extension Challenge:**
Measure actual latency difference between smem and gmem.

**Concepts:** On-chip memory, banks, latency

**Estimated Time:** 15-20 minutes

---

### Exercise 02: Bank Conflict Analysis
**File:** `ex02_bank_conflict_analysis.cu`

**Learning Objectives:**
- Identify bank conflict causes
- Analyze access patterns
- Calculate conflict severity
- Understand serialization impact

**Step-by-Step Guidance:**
1. **Create access pattern** - Row or column access
2. **Calculate bank IDs** - For each thread
3. **Detect conflicts** - Multiple threads, same bank
4. **Measure impact** - Performance degradation

**Key Concepts:**
- **Bank Conflict:** Multiple addresses map to same bank
- **Serialization:** Conflicting accesses happen sequentially
- **Severity:** 2-way, 4-way, 32-way conflict

**Bank Calculation:**
```cpp
// For 32-bit words (4 bytes)
bank = (byte_address / 4) % 32

// For float array
bank = (float_index) % 32
```

**Conflict Examples:**
```
Column access in 32×32 row-major matrix:
Thread 0: offset 0   → bank 0
Thread 1: offset 32  → bank 8  ✓
Thread 2: offset 64  → bank 16 ✓
Thread 3: offset 96  → bank 24 ✓
Thread 4: offset 128 → bank 0  ✗ CONFLICT with Thread 0!

This is a 4-way bank conflict (slows down by 4×)
```

**Common Pitfalls:**
- Column access in row-major layouts
- Power-of-2 strides causing conflicts
- Not analyzing both read and write patterns

**Expected Output:**
Bank conflict analysis with severity calculation.

**Verification:**
- Conflicts should be detected correctly
- Severity should match expected slowdown

**Extension Challenge:**
Design access pattern with zero conflicts.

**Concepts:** Conflicts, serialization, analysis

**Estimated Time:** 20-25 minutes

---

### Exercise 03: Padding for Conflict Avoidance
**File:** `ex03_padding_conflict_avoidance.cu`

**Learning Objectives:**
- Understand how padding works
- Calculate padding requirements
- Measure memory overhead trade-off
- Apply padding in kernels

**Step-by-Step Guidance:**
1. **Identify conflict pattern** - Find problematic access
2. **Add padding** - Increase stride by 1
3. **Verify conflict-free** - Check bank distribution
4. **Calculate overhead** - Memory cost

**Key Concepts:**
- **Padding:** Extra elements to change stride
- **Trade-off:** Memory overhead for performance
- **Optimal Padding:** Minimum padding for conflict-free

**Padding Example:**
```
Without padding (32×32):
Row stride = 32 → column access causes 32-way conflict

With padding (32×33):
Row stride = 33 → column access is conflict-free!
Overhead = 1/32 = 3.125%
```

**Common Pattern:**
```cpp
// Padded shared memory
__shared__ float smem[32][33];  // +1 padding per row

// Access
smem[threadIdx.y][threadIdx.x] = value;
```

**Padding Guidelines:**
| Matrix Size | Padded Stride | Overhead |
|-------------|---------------|----------|
| 32×32 | 33 | 3.1% |
| 64×64 | 65 | 1.6% |
| 128×128 | 129 | 0.8% |

**Common Pitfalls:**
- Too much padding (wasteful)
- Too little padding (still conflicts)
- Forgetting to update loop bounds

**Expected Output:**
Padded layout with conflict analysis.

**Verification:**
- No bank conflicts after padding
- Overhead should match calculation

**Extension Challenge:**
Calculate optimal padding for 16×64 matrix.

**Concepts:** Padding, stride, overhead

**Estimated Time:** 20-25 minutes

---

### Exercise 04: Swizzling Fundamentals
**File:** `ex04_swizzling_fundamentals.cu`

**Learning Objectives:**
- Understand swizzling concepts
- Learn XOR-based transformation
- See address remapping
- Compare with padding

**Step-by-Step Guidance:**
1. **Understand swizzling** - XOR-based address mapping
2. **Apply XOR transform** - Remap addresses
3. **Verify distribution** - Check bank spread
4. **Compare with padding** - Pros and cons

**Key Concepts:**
- **Swizzling:** XOR-based address transformation
- **Zero Overhead:** No extra memory needed
- **Reversible:** Can undo the transformation

**XOR Swizzling:**
```cpp
// Simple XOR swizzle (5-bit for 32 banks)
__device__ __forceinline__ int swizzle(int addr) {
    return addr ^ (addr >> 5);
}

// Unswizzle (same operation - XOR is reversible!)
__device__ __forceinline__ int unswizzle(int swizzled) {
    return swizzled ^ (swizzled >> 5);
}
```

**Swizzling vs Padding:**
| Aspect | Padding | Swizzling |
|--------|---------|-----------|
| Memory | +3% overhead | 0% overhead |
| Complexity | Simple | Moderate |
| Flexibility | Fixed pattern | Configurable |

**Common Pitfalls:**
- Incorrect XOR mask
- Not applying to both load and store
- Confusing logical vs physical addresses

**Expected Output:**
Swizzling transformation with bank distribution.

**Verification:**
- Swizzled addresses should spread across banks
- Original data should be recoverable

**Extension Challenge:**
Implement 2-level XOR swizzling.

**Concepts:** Swizzling, XOR, remapping

**Estimated Time:** 25-30 minutes

---

### Exercise 05: XOR-Based Swizzling
**File:** `ex05_xor_swizzling.cu`

**Learning Objectives:**
- Master XOR swizzling patterns
- Understand XOR properties
- Learn common patterns
- Implement multi-bit swizzling

**Step-by-Step Guidance:**
1. **Learn XOR properties** - Reversible, self-inverse
2. **Apply single-bit swizzle** - XOR with one bit
3. **Apply multi-bit swizzle** - XOR multiple bits
4. **Verify reversibility** - unswizzle(swizzle(x)) = x

**Key Concepts:**
- **XOR Properties:** A ⊕ B ⊕ B = A (self-inverse)
- **Bit Selection:** Which bits to XOR matters
- **Multi-bit:** Combine multiple XOR operations

**Common Swizzle Patterns:**
```cpp
// 5-bit swizzle (for 32 banks)
int swizzle_5bit(int addr) {
    return addr ^ (addr >> 5);
}

// Combined swizzle (more thorough mixing)
int swizzle_combined(int addr) {
    addr = addr ^ (addr >> 5);
    addr = addr ^ (addr >> 3);
    return addr;
}
```

**XOR Truth Table:**
```
A | B | A ⊕ B
0 | 0 |   0
0 | 1 |   1
1 | 0 |   1
1 | 1 |   0
```

**Common Pitfalls:**
- Wrong shift amount
- Not verifying reversibility
- Using non-reversible operations

**Expected Output:**
XOR swizzling with reversibility verification.

**Verification:**
- unswizzle(swizzle(x)) should equal x
- Banks should be evenly distributed

**Extension Challenge:**
Design custom XOR pattern for specific access pattern.

**Concepts:** XOR, bit manipulation, reversibility

**Estimated Time:** 25-30 minutes

---

### Exercise 06: Shared Memory Layouts for GEMM
**File:** `ex06_smem_layouts_gemm.cu`

**Learning Objectives:**
- Design GEMM shared memory layouts
- Understand tile A and B requirements
- Optimize access patterns
- Choose padding vs swizzling

**Step-by-Step Guidance:**
1. **Analyze GEMM access** - How A and B are accessed
2. **Design layout A** - For row access
3. **Design layout B** - For column access
4. **Apply optimization** - Padding or swizzling

**Key Concepts:**
- **GEMM Tiles:** Load A and B tiles to shared memory
- **Access Patterns:** A accessed by row, B by column
- **Optimization:** Both need conflict-free access

**GEMM Shared Memory Layout:**
```
For GEMM C = A × B:

Matrix A (M×K):
- Loaded by row
- Accessed by row in MMA
- Row-major layout works well

Matrix B (K×N):
- Loaded by column (problematic!)
- Accessed by column in MMA
- Needs padding or swizzling
```

**Common Pattern:**
```cpp
// Shared memory for GEMM tile
__shared__ float As[TILE_M][TILE_K];        // No padding needed
__shared__ float Bs[TILE_K][TILE_N + 1];    // +1 padding for B

// Or use swizzling
__shared__ float Bs_swizzled[TILE_K * TILE_N];
int physical_addr = swizzle(logical_addr);
```

**Common Pitfalls:**
- Not optimizing both A and B
- Wrong padding amount
- Ignoring MMA access pattern

**Expected Output:**
GEMM shared memory layout with optimization.

**Verification:**
- Both A and B should have conflict-free access
- Performance should improve

**Extension Challenge:**
Implement swizzled layout for B matrix.

**Concepts:** GEMM, tiles, optimization

**Estimated Time:** 30-35 minutes

---

## Learning Path

1. **Exercise 01** - Shared memory basics
2. **Exercise 02** - Bank conflict analysis
3. **Exercise 03** - Padding technique
4. **Exercise 04** - Swizzling fundamentals
5. **Exercise 05** - XOR swizzling
6. **Exercise 06** - GEMM layouts

## Bank Conflict Summary

### Conflict Scenarios

| Access Pattern | Stride | Conflict | Solution |
|----------------|--------|----------|----------|
| Row access | 1 | None | None needed |
| Column access | 32 | 32-way | Padding/Swizzle |
| Diagonal access | 33 | None | None needed |
| Transpose write | 32 | 32-way | Padding/Swizzle |

### Padding Options

| Matrix Size | Padded Stride | Overhead |
|-------------|---------------|----------|
| 32×32 | 33 | 3.1% |
| 64×64 | 65 | 1.6% |
| 128×128 | 129 | 0.8% |

## Key Formulas

### Bank Calculation
```
bank = (address / 4) % 32  // For 32-bit words
```

### XOR Swizzle
```
swizzled_addr = addr XOR (addr >> shift)
Common: shift = 5 for 32 banks
```

### Padding Overhead
```
overhead % = (padded_elements - original_elements) / original_elements × 100
```

## Tips for Success

1. **Analyze access patterns first** - Understand before optimizing
2. **Padding is simpler** - Use when memory allows
3. **Swizzling has no overhead** - Use when memory is tight
4. **Verify with analysis** - Always check bank distribution
5. **Consider both phases** - Load and compute must both be optimized

## Next Steps

After completing these exercises:
1. Move to Module 06: Collective Mainloops
2. Learn producer-consumer pipelines
3. Study complete kernel integration

## Additional Resources

- Module 05 README.md - Concept overview
- `shared_memory_layouts.cu` - Reference implementation
- CUDA Programming Guide - Shared Memory chapter
- CUTLASS documentation - https://github.com/NVIDIA/cutlass
