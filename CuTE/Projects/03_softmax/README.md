# Project 03: Softmax - Row-wise Reduction

## Objective

Implement a numerically stable row-wise softmax kernel using CuTe reduction patterns. This project teaches:
- Reduction operations in CUDA
- Numerical stability (max subtraction)
- Two-pass vs. online softmax algorithms
- Shared memory for intra-block reduction

## Theory

### The Problem

Given a matrix `X` of shape `(batch, seq_len)`, compute softmax row-wise:

```
softmax(x)[i] = exp(x[i] - max(x)) / Σ exp(x[j] - max(x))
```

The subtraction of `max(x)` ensures numerical stability by preventing overflow in `exp()`.

### Two-Pass Softmax

The naive approach requires two passes over the data:

```cuda
// Pass 1: Find maximum
float max_val = -infinity;
for (int i = 0; i < N; i++) {
    max_val = fmaxf(max_val, x[i]);
}

// Pass 2: Compute sum of exponentials
float sum = 0.0f;
for (int i = 0; i < N; i++) {
    sum += expf(x[i] - max_val);
}

// Pass 3: Normalize
for (int i = 0; i < N; i++) {
    y[i] = expf(x[i] - max_val) / sum;
}
```

### Online Softmax (Single Pass)

Online softmax computes the result in a single pass using running statistics:

```cpp
// Online softmax update
void online_update(float new_val, float& max_val, float& sum_exp) {
    if (new_val > max_val) {
        sum_exp *= expf(max_val - new_val);  // Rescale old sum
        max_val = new_val;
    }
    sum_exp += expf(new_val - max_val);
}
```

## Your Task

### Step 1: Understand Block-wise Reduction

Each thread block processes one row. Within the block, threads cooperate to:
1. Find the row maximum (reduction)
2. Compute sum of exponentials (reduction)
3. Normalize each element

```cpp
__global__ void softmax_cute(float* X, float* Y, int batch, int seq_len) {
    int row = blockIdx.x;
    if (row >= batch) return;
    
    // Each thread processes a subset of columns
    int tid = threadIdx.x;
    int stride = blockDim.x;
    
    // TODO: Implement reduction pattern
}
```

### Step 2: Implement Shared Memory Reduction

Use shared memory for efficient intra-block reduction:

```cpp
__shared__ float s_max[BLOCK_SIZE];
__shared__ float s_sum[BLOCK_SIZE];

// Each thread computes partial max/sum
s_max[tid] = thread_max;
__syncthreads();

// Reduce in log2 steps
for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
        s_max[tid] = fmaxf(s_max[tid], s_max[tid + s]);
    }
    __syncthreads();
}
```

### Step 3: Key CuTe Functions

| Function | Purpose | Example |
|----------|---------|---------|
| `make_layout(batch, seq_len)` | 2D row-major layout | `make_layout(B, N)` |
| `tiled_partition<32>(layout)` | Partition among warps | For warp-level reduction |
| `cute::reduce()` | Reduction operation | `reduce(tensor, op)` |

### Step 4: Implementation Hints

```cpp
__global__ void softmax_cute_kernel(float* X, float* Y, int batch, int seq_len) {
    // Each block handles one row
    int row = blockIdx.x;
    if (row >= batch) return;
    
    // Create 1D layout for this row
    auto row_layout = make_layout(seq_len);
    
    // Wrap pointers in tensors (offset to current row)
    float* row_X = X + row * seq_len;
    float* row_Y = Y + row * seq_len;
    auto tensor_X = make_tensor(make_gmem_ptr(row_X), row_layout);
    auto tensor_Y = make_tensor(make_gmem_ptr(row_Y), row_layout);
    
    // TODO: Implement 3-pass softmax
    // 1. Find max
    // 2. Compute sum of exp
    // 3. Normalize
}
```

## Exercises

### Exercise 1: Two-Pass Softmax (Required)

Implement the standard 3-kernel or 2-kernel approach:
- Kernel 1: Find row maximums
- Kernel 2: Compute sum of exponentials and normalize

### Exercise 2: Single-Block Softmax (Challenge)

Implement softmax where one thread block processes one entire row:
- Use shared memory for reductions
- Handle rows up to 1024 elements with 256 threads

### Exercise 3: Online Softmax (Advanced)

Implement the online softmax algorithm in a single kernel:
- Each thread maintains running (max, sum) state
- Use warp shuffles for efficient reduction
- Single pass over the data

## Verification

Your implementation is correct if:

```
[PASS] Softmax: All rows sum to 1.0 (max deviation: 0.000000)
Row 0 sum: 1.000000
Row 1 sum: 1.000000
Row 2 sum: 1.000000
Row 3 sum: 1.000000

Sample output (first row, first 8 elements):
  0.0321  0.0892  0.1543  0.2015  0.1876  0.1432  0.1021  0.0898
```

## Numerical Stability

### Why Subtract Max?

Without max subtraction:
```
exp(1000) → overflow (inf)
exp(-1000) → underflow (0)
```

With max subtraction:
```
x = [1000, 999, 998]
max = 1000
x - max = [0, -1, -2]
exp(x - max) = [1, 0.368, 0.135]  ✓ No overflow!
```

## Performance Notes

| Algorithm | Passes | Memory | Speed |
|-----------|--------|--------|-------|
| Naive 3-pass | 3 | High | Slow |
| Two-pass (fused) | 2 | Medium | Fast |
| Online softmax | 1 | Low | Fastest |

## Common Pitfalls

1. **Not handling empty rows**: Check `seq_len > 0`
2. **Division by zero**: Handle case where `sum_exp = 0`
3. **Race conditions**: Use proper reduction with `__syncthreads()`
4. **Incorrect stride**: Row stride = `seq_len` for row-major layout

## Mathematical Background

Softmax is defined as:

```
σ(x)ᵢ = exp(xᵢ) / Σⱼ exp(xⱼ)
```

The numerically stable version:

```
σ(x)ᵢ = exp(xᵢ - max(x)) / Σⱼ exp(xⱼ - max(x))
```

This is equivalent because:

```
exp(xᵢ - c) / Σⱼ exp(xⱼ - c) 
= exp(xᵢ) * exp(-c) / Σⱼ exp(xⱼ) * exp(-c)
= exp(xᵢ) / Σⱼ exp(xⱼ)
```

## Next Steps

After completing this project:
1. Try [Project 04: FlashAttention](../04_flash_attention/) which uses online softmax
2. Implement fused kernels (softmax + bias, softmax + dropout)
3. Explore block-softmax for very large sequences
4. Study how attention uses softmax

---

**Ready to normalize? Open `softmax.cu` and start implementing!**
