/*
 * Reductions Exercise
 *
 * This exercise demonstrates different approaches to implementing efficient reductions
 * in CUDA, including block-level and multi-block reductions.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Naive Reduction (INEFFICIENT)
__global__ void naiveReduction(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // Simply store the value - no actual reduction happening here
        output[tid] = input[tid];
    }
}

// Kernel 2: Optimized Block-Level Reduction
__global__ void blockLevelReduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Kernel 3: Student Exercise - Implement optimized reduction with warp-level optimizations
__global__ void studentOptimizedReduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // TODO: Implement optimized reduction with warp-level primitives for the final steps
    // HINT: Use warp shuffle operations for the last warp of the block to avoid sync overhead
    
    // Standard reduction for the first part
    for (int s = 1; s < blockDim.x/32; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // FIX: Use warp shuffle operations for the final warp-level reduction
    // This avoids the __syncthreads() overhead for the last few steps
    if (tid < 32) {
        // Perform warp-level reduction using shuffle
        /* YOUR WARP-LEVEL REDUCTION CODE */;
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Kernel 4: Student Exercise - Implement multi-block hierarchical reduction
__global__ void studentHierarchicalReduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform block-level reduction
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write partial result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
    
    // TODO: Implement the second phase of hierarchical reduction
    // HINT: Launch another kernel to reduce the partial results from each block
    // This would typically be done in a separate kernel call, but for this exercise,
    // consider how you might handle multiple phases in a single kernel
}

// Kernel 5: Student Exercise - Implement reduction for different operations (max, sum, etc.)
__global__ void studentMultiOpReduction(float* input, float* sum_output, float* max_output, int n, int op_type) {
    extern __shared__ float sdata_sum[];
    extern __shared__ float sdata_max[];  // This won't work - need to handle multiple arrays differently
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float mySum = (i < n) ? input[i] : 0.0f;
    float myMax = (i < n) ? input[i] : -FLT_MAX;  // For max operation
    
    // TODO: Implement reduction for both sum and max operations simultaneously
    // HINT: You'll need to handle the shared memory allocation differently
    // since extern __shared__ can't be declared twice
    
    // Store in shared memory
    sdata_sum[tid] = mySum;
    // sdata_max[tid] = myMax;  // This won't work - need alternative approach
    __syncthreads();
    
    // Perform reduction based on op_type
    if (op_type == 0) {  // Sum
        for (int s = 1; s < blockDim.x; s *= 2) {
            if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
                sdata_sum[tid] += sdata_sum[tid + s];
            }
            __syncthreads();
        }
        if (tid == 0) {
            sum_output[blockIdx.x] = sdata_sum[0];
        }
    }
    // TODO: Add implementation for max operation
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 100) * 0.1f;
    }
}

int main() {
    printf("=== Reductions Exercise ===\n");
    printf("Learn to implement efficient reduction operations in CUDA.\n\n");

    // Setup parameters
    const int N = 1024 * 16;  // Multiple of block size
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    size_t bytes = N * sizeof(float);
    size_t output_bytes = gridSize * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output_naive, *h_output_block, *h_output_opt, *h_output_hier;
    h_input = (float*)malloc(bytes);
    h_output_naive = (float*)malloc(bytes);  // Full size for naive
    h_output_block = (float*)malloc(output_bytes);
    h_output_opt = (float*)malloc(output_bytes);
    h_output_hier = (float*)malloc(output_bytes);
    
    // Initialize data
    initArray(h_input, N, 1.0f);
    
    // Allocate device memory
    float *d_input, *d_output_naive, *d_output_block, *d_output_opt, *d_output_hier;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output_naive, bytes);
    cudaMalloc(&d_output_block, output_bytes);
    cudaMalloc(&d_output_opt, output_bytes);
    cudaMalloc(&d_output_hier, output_bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Run naive reduction kernel
    printf("Running naive reduction kernel...\n");
    naiveReduction<<<gridSize, blockSize>>>(d_input, d_output_naive, N);
    cudaDeviceSynchronize();
    
    // Run block-level reduction kernel
    printf("Running block-level reduction kernel...\n");
    blockLevelReduction<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output_block, N);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student reduction exercises (complete the code first!)...\n");
    
    // Optimized reduction exercise
    studentOptimizedReduction<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output_opt, N);
    cudaDeviceSynchronize();
    
    // Hierarchical reduction exercise
    studentHierarchicalReduction<<<gridSize, blockSize, blockSize * sizeof(float)>>>(d_input, d_output_hier, N);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the reduction implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_naive, d_output_naive, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_block, d_output_block, output_bytes, cudaMemcpyDeviceToHost);
    
    // Calculate total from block results
    float total_block = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_block += h_output_block[i];
    }
    
    // Print sample results
    printf("\nSample results:\n");
    printf("Input sum (approx): %.2f\n", N * 1.0f + (N-1) * N * 0.1f / 2.0f);  // Approximate analytical sum
    printf("Block reduction total: %.2f\n", total_block);
    
    // Cleanup
    free(h_input); free(h_output_naive); free(h_output_block); 
    free(h_output_opt); free(h_output_hier);
    cudaFree(d_input); cudaFree(d_output_naive); cudaFree(d_output_block);
    cudaFree(d_output_opt); cudaFree(d_output_hier);
    
    printf("\nExercise completed! Notice how optimized reductions improve performance.\n");
    
    return 0;
}