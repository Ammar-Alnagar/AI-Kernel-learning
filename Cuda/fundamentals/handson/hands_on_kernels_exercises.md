# CUDA Hands-On Kernels Tutorial

## Overview

This hands-on tutorial provides practical exercises where you'll complete partially implemented CUDA kernels. Each exercise focuses on a fundamental CUDA concept with missing code sections marked with `TODO` comments. Your task is to fill in these sections to make the kernels compile and run correctly.

## Learning Objectives

By completing these exercises, you will:
- Practice essential CUDA programming patterns
- Understand thread indexing and memory access patterns
- Learn optimization techniques for memory access
- Gain hands-on experience with shared memory usage
- Develop proficiency in CUDA kernel implementation

## Exercise 1: Vector Addition

### Objective
Complete the vector addition kernel that computes `C = A + B`.

### Code to Complete
```cuda
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    // TODO: Calculate the global thread index
    // Hint: Use blockIdx.x, blockDim.x, and threadIdx.x
    int i = /* YOUR CODE HERE */;

    // TODO: Add bounds checking to prevent out-of-bounds access
    if (/* YOUR CONDITION HERE */) {
        // TODO: Perform the vector addition: C[i] = A[i] + B[i]
        /* YOUR CODE HERE */;
    }
}
```

### Solution Guidance
- Calculate the global thread index using the formula: `blockIdx.x * blockDim.x + threadIdx.x`
- Check that the calculated index is less than N to avoid out-of-bounds access
- Perform the element-wise addition operation

## Exercise 2: Matrix Multiplication

### Objective
Complete the matrix multiplication kernel that computes `C = A × B`.

### Code to Complete
```cuda
__global__ void matrixMul(float* A, float* B, float* C, int width) {
    // TODO: Calculate row and column indices for this thread
    int row = /* YOUR CODE HERE */;
    int col = /* YOUR CODE HERE */;

    // Only compute if within matrix bounds
    if (row < width && col < width) {
        float sum = 0.0f;
        
        // TODO: Compute dot product of row from A and column from B
        // Hint: Loop from 0 to width and accumulate the products
        for (int k = 0; k < width; k++) {
            // YOUR CODE HERE: multiply A[row][k] by B[k][col] and add to sum
            /* YOUR CODE HERE */;
        }
        
        // Store result in C[row][col]
        C[row * width + col] = sum;
    }
}
```

### Solution Guidance
- Calculate row index using block and thread Y coordinates
- Calculate column index using block and thread X coordinates
- Implement the dot product calculation by multiplying corresponding elements and accumulating the result

## Exercise 3: Reduction Operation

### Objective
Complete the reduction kernel that computes the sum of array elements.

### Code to Complete
```cuda
__global__ void reductionSum(float* input, float* output, int n) {
    // TODO: Declare shared memory for this block
    // Hint: Use __shared__ keyword and size it appropriately
    /* YOUR DECLARATION HERE */;

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory
    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0.0f;  // Pad with zeros
    }
    __syncthreads();

    // Perform reduction in shared memory
    // TODO: Complete the reduction loop
    // Hint: Each iteration reduces the number of active elements by half
    for (int s = 1; s < blockDim.x; s *= 2) {
        // TODO: Check bounds and perform reduction
        if (/* YOUR CONDITION HERE */) {
            // TODO: Add element at tid+s to element at tid
            /* YOUR CODE HERE */;
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}
```

### Solution Guidance
- Declare shared memory array with size equal to blockDim.x
- In the reduction loop, check if tid+s is within block bounds
- Add the element at sdata[tid+s] to sdata[tid]
- Use `__syncthreads()` to synchronize threads after each step

## Exercise 4: Memory Coalescing

### Objective
Fix the memory access pattern to ensure coalesced access for optimal performance.

### Code to Complete
```cuda
__global__ void coalescedCopy(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Implement coalesced memory access
    // Current implementation has poor coalescing - fix it
    if (tid < n) {
        // Implement proper coalesced access where consecutive threads
        // access consecutive memory locations
        /* YOUR CODE HERE */;
    }
}
```

### Solution Guidance
- Ensure that consecutive threads access consecutive memory addresses
- The simplest coalesced access is `output[tid] = input[tid]`

## Exercise 5: Shared Memory Transpose

### Objective
Fix bank conflicts in the shared memory transpose operation.

### Code to Complete
```cuda
__global__ void sharedMemoryTranspose(float* input, float* output, int width) {
    // TODO: Modify shared memory declaration to avoid bank conflicts
    // Hint: Add padding to avoid bank conflicts during transposed access
    __shared__ float tile[32][33];  // Modified to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    // Load data into shared memory (coalesced read)
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Corrected transposed write that avoids bank conflicts
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}
```

### Solution Guidance
- Add padding to the shared memory declaration (e.g., `[32][33]` instead of `[32][32]`)
- This prevents multiple threads from accessing the same memory bank simultaneously
- The padding breaks the alignment that causes bank conflicts during transposed access

## Compilation and Testing

To compile and test your implementations:

```bash
nvcc hands_on_kernels_tutorial.cu -o hands_on_kernels_tutorial
./hands_on_kernels_tutorial
```

## Verification

After completing each exercise, verify your implementation by:
1. Ensuring the code compiles without errors
2. Checking that the output matches expected results
3. Comparing performance characteristics with optimized implementations

## Additional Challenges

Once you've completed all exercises, try these advanced challenges:
1. Optimize the reduction kernel further using different strategies
2. Implement memory access patterns for 3D matrices
3. Add error checking to all CUDA API calls
4. Profile your kernels using NVIDIA Nsight Compute to identify bottlenecks