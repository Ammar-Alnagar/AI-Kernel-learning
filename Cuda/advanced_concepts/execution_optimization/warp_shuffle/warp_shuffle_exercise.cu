/*
 * Warp Shuffle Exercise
 *
 * This exercise demonstrates how to use warp shuffle operations for efficient
 * intra-warp communication without using shared memory.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Using Shared Memory for Communication (LESS EFFICIENT)
__global__ void sharedMemCommunication(float* input, float* output, int n) {
    __shared__ float sdata[32];  // One warp's data
    
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;
    
    if (tid < n && threadIdx.x < 32) {
        // Load data into shared memory
        sdata[laneId] = (tid < n) ? input[tid] : 0.0f;
    }
    __syncthreads();
    
    if (tid < n && threadIdx.x < 32) {
        // Get next thread's value using shared memory
        float nextVal = sdata[(laneId + 1) % 32];
        output[tid] = nextVal;
    }
}

// Kernel 2: Using Warp Shuffle for Communication (MORE EFFICIENT)
__global__ void shuffleCommunication(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;
    
    if (tid < n) {
        float value = input[tid];
        
        // Get next thread's value using warp shuffle
        float nextVal = __shfl_up_sync(0xFFFFFFFF, value, 1, 32);
        
        // If this is the first thread in the warp, use the last thread's value
        if (laneId == 0) {
            float lastVal = __shfl_sync(0xFFFFFFFF, value, 31, 32);
            output[tid] = lastVal;
        } else {
            output[tid] = nextVal;
        }
    }
}

// Kernel 3: Student Exercise - Implement prefix sum using warp shuffle
__global__ void studentWarpPrefixSum(float* input, float* output, int n) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;
    
    if (tid < n) {
        float value = input[tid];
        
        // TODO: Implement warp-level prefix sum using shuffle operations
        // HINT: Use __shfl_up_sync to get values from previous lanes and accumulate
        
        float result = /* YOUR PREFIX SUM IMPLEMENTATION */;
        
        output[tid] = result;
    }
}

// Kernel 4: Student Exercise - Implement warp broadcast and reduction
__global__ void studentWarpOperations(float* input, float* output, int n, int* indices) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    int laneId = threadIdx.x % 32;
    int warpId = threadIdx.x / 32;
    
    if (tid < n) {
        float value = input[tid];
        int idx = indices[tid % 32];  // Use only first 32 indices
        
        // TODO: Implement warp broadcast - broadcast value from lane 'idx' to all lanes
        // HINT: Use __shfl_sync to broadcast a value from a specific lane
        float broadcasted = /* YOUR BROADCAST IMPLEMENTATION */;
        
        // TODO: Implement warp reduction - compute sum of all values in the warp
        // HINT: Use repeated __shfl_down_sync operations to accumulate
        float warpSum = /* YOUR REDUCTION IMPLEMENTATION */;
        
        // Store results
        output[tid] = broadcasted + warpSum;
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + i * 0.1f;
    }
}

// Utility function to initialize indices
void initIndices(int* indices, int n) {
    for (int i = 0; i < n; i++) {
        indices[i] = i % 32;  // Cycle through lane IDs
    }
}

int main() {
    printf("=== Warp Shuffle Exercise ===\n");
    printf("Learn to use warp shuffle operations for efficient intra-warp communication.\n\n");

    // Setup parameters
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    size_t int_bytes = 32 * sizeof(int);  // Only need 32 indices for one warp
    
    // Allocate host memory
    float *h_input, *h_output_shared, *h_output_shuffle, *h_output_prefix, *h_output_ops;
    int *h_indices;
    h_input = (float*)malloc(bytes);
    h_output_shared = (float*)malloc(bytes);
    h_output_shuffle = (float*)malloc(bytes);
    h_output_prefix = (float*)malloc(bytes);
    h_output_ops = (float*)malloc(bytes);
    h_indices = (int*)malloc(int_bytes);
    
    // Initialize data
    initArray(h_input, N, 1.0f);
    initIndices(h_indices, 32);
    
    // Allocate device memory
    float *d_input, *d_output_shared, *d_output_shuffle, *d_output_prefix, *d_output_ops;
    int *d_indices;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output_shared, bytes);
    cudaMalloc(&d_output_shuffle, bytes);
    cudaMalloc(&d_output_prefix, bytes);
    cudaMalloc(&d_output_ops, bytes);
    cudaMalloc(&d_indices, int_bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, int_bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Run shared memory communication kernel
    printf("Running shared memory communication kernel...\n");
    sharedMemCommunication<<<gridSize, blockSize>>>(d_input, d_output_shared, N);
    cudaDeviceSynchronize();
    
    // Run shuffle communication kernel
    printf("Running warp shuffle communication kernel...\n");
    shuffleCommunication<<<gridSize, blockSize>>>(d_input, d_output_shuffle, N);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student warp shuffle exercises (complete the code first!)...\n");
    
    // Prefix sum exercise
    studentWarpPrefixSum<<<gridSize, blockSize>>>(d_input, d_output_prefix, N);
    cudaDeviceSynchronize();
    
    // Warp operations exercise
    studentWarpOperations<<<gridSize, blockSize>>>(d_input, d_output_ops, N, d_indices);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the warp shuffle implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_shared, d_output_shared, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_shuffle, d_output_shuffle, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_prefix, d_output_prefix, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_ops, d_output_ops, bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Input:        %.2f %.2f %.2f %.2f %.2f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    printf("Shared Mem:   %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_shared[0], h_output_shared[1], h_output_shared[2], h_output_shared[3], h_output_shared[4]);
    printf("Warp Shuffle: %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_shuffle[0], h_output_shuffle[1], h_output_shuffle[2], h_output_shuffle[3], h_output_shuffle[4]);
    
    // Cleanup
    free(h_input); free(h_output_shared); free(h_output_shuffle); 
    free(h_output_prefix); free(h_output_ops); free(h_indices);
    cudaFree(d_input); cudaFree(d_output_shared); cudaFree(d_output_shuffle);
    cudaFree(d_output_prefix); cudaFree(d_output_ops); cudaFree(d_indices);
    
    printf("\nExercise completed! Notice how warp shuffles can replace shared memory for simple communication.\n");
    
    return 0;
}