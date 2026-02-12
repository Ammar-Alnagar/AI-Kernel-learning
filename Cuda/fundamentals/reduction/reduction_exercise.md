# Reduction Operation Hands-On Exercise

## Objective
Complete the reduction kernel that computes the sum of array elements.

## Code to Complete
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

## Solution Guidance
- Declare shared memory array with size equal to blockDim.x: `__shared__ float sdata[256];` (assuming max block size of 256)
- In the reduction loop, check if `tid + s < blockDim.x` to stay within bounds
- Add the element at `sdata[tid+s]` to `sdata[tid]`: `sdata[tid] += sdata[tid + s];`
- Use `__syncthreads()` to synchronize threads after each step

## Key Concepts Practiced
- Shared memory usage
- Parallel reduction algorithms
- Thread synchronization
- Memory coalescing in loading phase

## Verification
After completing the kernel:
1. Ensure the code compiles without errors
2. Check that the partial sums computed by each block are correct
3. Verify that the final sum matches the expected total
4. Confirm that the algorithm scales properly with different array sizes