# Project 01: Vector Add - Introduction to CuTe

## Objective

Implement a vector addition kernel (`C = A + B`) using CuTe abstractions. This project introduces:
- Creating CuTe tensors from raw pointers
- Mapping threads to work using layouts
- Element-wise operations with proper memory access patterns

## Theory

### The Problem

Given two vectors `A` and `B` of size `N`, compute `C = A + B`.

While simple, this kernel teaches fundamental concepts:
1. **Thread indexing**: How to map CUDA threads to data elements
2. **Memory coalescing**: Ensuring consecutive threads access consecutive memory
3. **CuTe abstractions**: Using layouts instead of manual index calculations

### Traditional CUDA Approach

```cuda
__global__ void vector_add_cuda(float* A, float* B, float* C, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}
```

### CuTe Approach

CuTe provides algebraic abstractions for indexing:

```cpp
// Define a layout: 1D shape with stride-1
auto layout = make_layout(N);

// Wrap pointers in tensors
auto tensor_A = make_tensor(make_gmem_ptr(A), layout);
auto tensor_B = make_tensor(make_gmem_ptr(B), layout);
auto tensor_C = make_tensor(make_gmem_ptr(C), layout);

// Get thread's work using partitioning
auto thread_layout = make_layout(blockDim.x);
auto thread_idx = threadIdx.x;
```

## Your Task

### Step 1: Understand the Setup Code

The provided `vector_add.cu` contains:
- Host-side memory allocation and initialization
- Kernel launch configuration
- Verification code

Your job is to implement the **device kernel** using CuTe.

### Step 2: Implement the CuTe Kernel

In `vector_add.cu`, find the `// TODO:` comments and complete:

1. **Create layouts** for the vectors
2. **Wrap pointers** in CuTe tensors
3. **Partition work** among threads
4. **Implement the element-wise add** loop

### Step 3: Key CuTe Functions

You'll need these functions (all in `cute/layout.hpp` and `cute/tensor.hpp`):

| Function | Purpose | Example |
|----------|---------|---------|
| `make_layout(shape)` | Create a layout | `make_layout(1024)` |
| `make_tensor(ptr, layout)` | Wrap pointer in tensor | `make_tensor(ptr, layout)` |
| `make_gmem_ptr(ptr)` | Cast to GMEM pointer | `make_gmem_ptr(raw_ptr)` |
| `get<0>(layout)` | Get shape dimension | `get<0>(layout)` |
| `thread_idx` | Thread's index | Used for partitioning |

### Step 4: Implementation Hints

```cpp
__global__ void vector_add_cute(float* A, float* B, float* C, int N) {
    // TODO 1: Create a 1D layout for the vectors
    // Hint: auto layout = make_layout(N);
    
    // TODO 2: Wrap raw pointers in CuTe tensors
    // Hint: Use make_tensor(make_gmem_ptr(...), layout)
    
    // TODO 3: Determine which elements this thread processes
    // Hint: Use threadIdx.x and blockDim.x to calculate stride
    
    // TODO 4: Loop through assigned elements and compute C[i] = A[i] + B[i]
    // Hint: for (int i = thread_idx; i < N; i += stride)
}
```

## Exercises

### Exercise 1: Basic Implementation (Required)

Complete the kernel to handle any vector size `N`. Each thread should process multiple elements if `N > num_threads`.

### Exercise 2: Vectorized Access (Challenge)

Modify your implementation to use `float4` (128-bit) loads when possible:

```cpp
// TODO: Check if pointer is aligned and N is multiple of 4
// Use reinterpret_cast<float4*> for vectorized access
```

### Exercise 3: Tiled Approach (Advanced)

Process elements in tiles using CuTe's tiling utilities:

```cpp
// TODO: Create a tile of 128 elements
// Use cute::tiled_partition to distribute work
```

## Verification

Your implementation is correct if:

```
[PASS] Vector Add: All elements match (max error: 0.000000)
Vector A[0:5]: 1.000 2.000 3.000 4.000 5.000
Vector B[0:5]: 10.000 20.000 30.000 40.000 50.000
Vector C[0:5]: 11.000 22.000 33.000 44.000 55.000
```

## Performance Notes

For a kernel this simple, CuTe adds minimal overhead but provides:
- **Composability**: Easy to extend to multi-dimensional tensors
- **Type safety**: Compile-time shape checking
- **Abstraction**: Same code works for different layouts

## Common Pitfalls

1. **Forgetting bounds checks**: Always verify `idx < N`
2. **Wrong stride calculation**: Thread stride = `gridDim.x * blockDim.x`
3. **Not using make_gmem_ptr**: CuTe needs typed pointers

## Next Steps

After completing this project:
1. Try [Project 02: GEMM](../02_gemm/) for matrix operations
2. Experiment with 2D layouts (matrices instead of vectors)
3. Explore CuTe's `tiled_copy` for more efficient memory access

---

**Ready to code? Open `vector_add.cu` and start implementing!**
