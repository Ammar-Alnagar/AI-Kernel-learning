# Module 02: CuTe Tensors - Exercises

## Overview
This directory contains hands-on exercises to practice CuTe Tensor concepts. Tensors wrap raw pointers with layouts to create safe, indexed views of memory.

## Building the Exercises

### Prerequisites
- CUDA Toolkit with sm_89 support
- CUTLASS library with CuTe headers
- Complete Module 01 exercises (Layout Algebra)

### Build Instructions

```bash
cd exercises
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
```

### Running Exercises

```bash
./ex01_tensor_creation
./ex02_tensor_access
```

---

## Exercises

### Exercise 01: Tensor Creation from Raw Pointers
**File:** `ex01_tensor_creation.cu`

**Learning Objectives:**
- Create CuTe tensors from raw pointers
- Wrap pointers with layouts using `make_tensor`
- Access tensor elements using coordinates
- Work with different data types (float, int, half)

**Step-by-Step Guidance:**
1. **Task 1:** Create a 1D tensor from raw array
   - Allocate array: `float data[16]`
   - Create layout: `make_layout(make_shape(Int<16>{}), GenRowMajor{})`
   - Make tensor: `make_tensor(data, layout)`

2. **Task 2:** Create a 2D row-major tensor (4×4)
   - Use `make_shape(Int<4>{}, Int<4>{})`
   - Apply `GenRowMajor{}`

3. **Task 3:** Create tensors with different types
   - Try `int`, `double`, `cutlass::half_t`

4. **Task 4:** Access elements using coordinates
   - Use `tensor(i, j)` syntax

**Key Concepts:**
- **Tensor = Pointer + Layout**
- `make_tensor(ptr, layout)` creates the tensor
- Access via `tensor(coord...)` not pointer arithmetic
- Type safety through template parameters

**Common Pattern:**
```cpp
float* raw_ptr = ...;
auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
auto tensor = make_tensor(raw_ptr, layout);
float val = tensor(2, 3);  // Access element
```

**Common Pitfalls:**
- Forgetting to match layout shape to data size
- Using wrong coordinate order (row, col) vs (col, row)
- Not initializing data before access

**Expected Output:**
Tensor structures showing shape, stride, and data pointer.

**Verification:**
- Tensor size should match layout size
- Element access should return expected values

**Extension Challenge:**
Create a tensor with custom stride layout and verify access.

**Concepts:** `make_tensor`, pointer wrappers, element access

**Estimated Time:** 15-20 minutes

---

### Exercise 02: Tensor Access Patterns
**File:** `ex02_tensor_access.cu`

**Learning Objectives:**
- Understand row-wise vs column-wise access
- Identify coalesced vs uncoalesced access patterns
- Measure impact on memory throughput
- Choose appropriate layouts for access patterns

**Step-by-Step Guidance:**
1. **Create row-major tensor** - 8×8 matrix
2. **Access row-wise** - Iterate columns in inner loop
3. **Access column-wise** - Iterate rows in inner loop
4. **Compare performance** - Notice the difference

**Key Concepts:**
- **Coalesced Access:** Consecutive threads access consecutive addresses
- **Uncoalesced Access:** Scattered memory access (slow)
- **Memory Throughput:** Bytes/second achieved

**Access Pattern Example:**
```cpp
// Row-wise (coalesced for row-major)
for (int i = 0; i < M; ++i)
    for (int j = 0; j < N; ++j)
        val = tensor(i, j);  // Good!

// Column-wise (uncoalesced for row-major)
for (int j = 0; j < N; ++j)
    for (int i = 0; i < M; ++i)
        val = tensor(i, j);  // Bad for row-major!
```

**Common Pitfalls:**
- Mismatched layout and access pattern
- Not considering access pattern when designing layout
- Ignoring performance implications

**Expected Output:**
Access pattern analysis showing coalesced vs uncoalesced behavior.

**Verification:**
- Row-wise access on row-major should show stride-1 pattern
- Column-wise access should show larger strides

**Extension Challenge:**
Create column-major layout and verify column-wise access is now coalesced.

**Concepts:** Coalescing, memory efficiency, access patterns

**Estimated Time:** 20-25 minutes

---

### Exercise 03: Tensor Slicing Operations
**File:** `ex03_tensor_slicing.cu`

**Learning Objectives:**
- Extract sub-tensors and views
- Create row and column slices
- Extract sub-matrices
- Perform strided slices

**Step-by-Step Guidance:**
1. **Create source tensor** - 8×8 matrix
2. **Extract row slice** - Get row 3 as 1D view
3. **Extract column slice** - Get column 5 as 1D view
4. **Extract sub-matrix** - Get rows 2-5, cols 2-5

**Key Concepts:**
- **View:** No data copy, just different interpretation
- **Slice:** Extract subset of dimensions
- **Zero-copy:** Slicing doesn't duplicate data

**Common Pattern:**
```cpp
// Extract row i (conceptually)
auto row_i = tensor(i, _);  // All columns of row i

// Extract column j
auto col_j = tensor(_, j);  // All rows of column j

// Extract sub-region (conceptual)
auto sub = tensor(make_range(2, 6), make_range(2, 6));
```

**Common Pitfalls:**
- Assuming slice creates a copy (it doesn't)
- Modifying slice affects original tensor
- Out-of-bounds slice ranges

**Expected Output:**
Original tensor and various slices showing the views.

**Verification:**
- Slice size should match expected dimensions
- Slice data should match original tensor region

**Extension Challenge:**
Create a strided slice (every other row).

**Concepts:** Slicing, views, no-copy operations

**Estimated Time:** 20-25 minutes

---

### Exercise 04: Tensor Transpose and View
**File:** `ex04_tensor_transpose.cu`

**Learning Objectives:**
- Create transposed views without copying data
- Verify transpose relationships
- Understand double transpose
- Use transpose in algorithms

**Step-by-Step Guidance:**
1. **Create original tensor** - 4×6 matrix
2. **Create transposed view** - 6×4 without copying
3. **Verify relationship** - A(i,j) = A^T(j,i)
4. **Double transpose** - Verify (A^T)^T = A

**Key Concepts:**
- **Transpose View:** Reinterpret layout, don't move data
- **Zero-copy:** Transpose is free (just layout change)
- **Logical vs Physical:** Data stays, view changes

**Visual Example:**
```
Original A (2×3):     Transposed A^T (3×2):
1 2 3                 1 4
4 5 6                 2 5
                      3 6

Same data, different layout interpretation!
```

**Common Pitfalls:**
- Expecting data to be rearranged (it's not)
- Confusing transpose view with transpose copy
- Forgetting stride changes in transpose

**Expected Output:**
Original and transposed tensors showing the relationship.

**Verification:**
- A(i,j) should equal A_transposed(j,i)
- Double transpose should equal original

**Extension Challenge:**
Use transpose view in a matrix multiplication.

**Concepts:** Transpose, view operations, zero-copy

**Estimated Time:** 15-20 minutes

---

### Exercise 05: Tensor Composition with Layouts
**File:** `ex05_tensor_composition.cu`

**Learning Objectives:**
- Compose tensors hierarchically
- Understand tile and element layouts
- Create multi-level organization
- Design tiled algorithms

**Step-by-Step Guidance:**
1. **Create tile layout** - How tiles are arranged (2×2)
2. **Create element layout** - Elements per tile (4×4)
3. **Compose layouts** - Combine into hierarchical layout
4. **Create composed tensor** - Apply to data

**Key Concepts:**
- **Composition:** Building complex from simple
- **Tile Layout:** Organization of tiles
- **Element Layout:** Organization within tiles

**Common Pattern:**
```cpp
// Tile layout: 2×2 tiles
auto tile_layout = make_layout(make_shape(Int<2>{}, Int<2>{}), ...);

// Element layout: 4×4 elements per tile
auto elem_layout = make_layout(make_shape(Int<4>{}, Int<4>{}), ...);

// Composed: 8×8 total (2×4 × 2×4)
auto composed_layout = composition(tile_layout, elem_layout);
```

**Common Pitfalls:**
- Mismatched tile and element sizes
- Not understanding hierarchical indexing
- Incorrect composition order

**Expected Output:**
Hierarchical tensor showing tile and element organization.

**Verification:**
- Total size = tiles × elements_per_tile
- Indexing should work at both levels

**Extension Challenge:**
Create 3-level hierarchy: Block → Tile → Element.

**Concepts:** Composition, tiling, hierarchy

**Estimated Time:** 25-30 minutes

---

### Exercise 06: Multi-dimensional Tensors
**File:** `ex06_multidim_tensors.cu`

**Learning Objectives:**
- Work with 3D tensors (volumes)
- Create 4D tensors for batches (NCHW)
- Calculate strides for multi-dimensional layouts
- Understand tensor formats (NCHW vs NHWC)

**Step-by-Step Guidance:**
1. **Create 3D tensor** - Volume (4×4×4)
2. **Create 4D tensor** - Batch (2×3×4×4) = (N×C×H×W)
3. **Calculate strides** - Understand stride pattern
4. **Compare formats** - NCHW vs NHWC

**Key Concepts:**
- **3D Tensor:** Volume data, RGB images
- **4D Tensor:** Batched data for ML
- **NCHW:** Batch, Channel, Height, Width (PyTorch default)
- **NHWC:** Batch, Height, Width, Channel (TensorFlow default)

**Stride Calculation:**
```
For NCHW (2×3×4×4):
Stride = (48, 16, 4, 1)  // N×C×H×W order
```

**Common Pitfalls:**
- Confusing dimension order
- Wrong stride calculation
- Mixing NCHW and NHWC

**Expected Output:**
Multi-dimensional tensors with stride information.

**Verification:**
- Total size = product of all dimensions
- Stride pattern should match format

**Extension Challenge:**
Convert between NCHW and NHWC layouts.

**Concepts:** Multi-dimensional, NCHW/NHWC, stride calculation

**Estimated Time:** 25-30 minutes

---

### Exercise 07: Tensor Memory Spaces
**File:** `ex07_tensor_memory_spaces.cu`

**Learning Objectives:**
- Understand CUDA memory spaces (gmem, smem, rmem)
- Create tensors in different memory spaces
- Use pointer wrappers correctly
- Move data between memory spaces

**Step-by-Step Guidance:**
1. **Global memory tensor** - Use `make_gmem_ptr`
2. **Shared memory tensor** - Use `make_smem_ptr`
3. **Register memory tensor** - Use `make_rmem_ptr`
4. **Compare characteristics** - Size, latency, scope

**Key Concepts:**
- **Global (gmem):** Large, slow, all threads
- **Shared (smem):** Small, fast, block scope
- **Register (rmem):** Tiny, fastest, thread scope

**Memory Space Comparison:**
| Property | Global | Shared | Register |
|----------|--------|--------|----------|
| Size | GBs | KBs/MBs | KBs |
| Latency | High | Low | Lowest |
| Scope | All | Block | Thread |

**Common Pattern:**
```cpp
// Global memory
float* gmem_data;
cudaMalloc(&gmem_data, size);
auto gmem_tensor = make_tensor(make_gmem_ptr(gmem_data), layout);

// Shared memory
extern __shared__ float smem[];
auto smem_tensor = make_tensor(make_smem_ptr(smem), layout);

// Register memory
float rmem[16];
auto rmem_tensor = make_tensor(make_rmem_ptr(rmem), layout);
```

**Common Pitfalls:**
- Using wrong pointer wrapper
- Exceeding shared memory limits
- Not synchronizing shared memory access

**Expected Output:**
Tensors in different memory spaces with characteristics.

**Verification:**
- Each tensor should use correct pointer type
- Memory space should match intended use

**Extension Challenge:**
Implement gmem → smem → rmem data movement.

**Concepts:** Memory hierarchy, pointer wrappers, data movement

**Estimated Time:** 25-30 minutes

---

## Learning Path

1. **Exercise 01** - Create basic tensors
2. **Exercise 02** - Access patterns matter
3. **Exercise 03** - Slice tensors
4. **Exercise 04** - Transpose views
5. **Exercise 05** - Compose layouts
6. **Exercise 06** - Multi-dimensional
7. **Exercise 07** - Memory spaces

## Tips for Success

1. **Match layout to access pattern** for coalescing
2. **Use views** instead of copying when possible
3. **Understand memory hierarchy** for optimization
4. **Broadcast** to avoid data duplication
5. **Print tensors** to understand their structure

## Common Patterns Reference

```cpp
// Create tensor from raw pointer
float* data = ...;
auto layout = make_layout(make_shape(Int<4>{}, Int<4>{}), GenRowMajor{});
auto tensor = make_tensor(make_gmem_ptr(data), layout);

// Access elements
float val = tensor(i, j);

// Create shared memory tensor
extern __shared__ float smem[];
auto smem_tensor = make_tensor(make_smem_ptr(smem), layout);

// Create register tensor
float rmem[16];
auto rmem_tensor = make_tensor(make_rmem_ptr(rmem), layout);

// Broadcast layout (stride = 0)
auto broadcast = make_layout(shape, make_stride(Int<0>{}, Int<1>{}));
```

## Next Steps

After completing these exercises:
1. Move to Module 03: Tiled Copy
2. Learn cooperative thread operations
3. Study vectorized memory transfers

## Additional Resources

- Module 02 README.md - Concept overview
- `tensor_basics.cu` - Reference implementation
- CuTe documentation - https://github.com/NVIDIA/cutlass
- CUTLASS examples - https://github.com/NVIDIA/cutlass/tree/master/examples
