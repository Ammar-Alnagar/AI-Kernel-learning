/*
 * Thread Coarsening Exercise
 *
 * This exercise demonstrates how to use thread coarsening to reduce kernel launch overhead
 * and improve occupancy by having each thread process multiple elements.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Fine-grained threading (ONE ELEMENT PER THREAD)
__global__ void fineGrainedKernel(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Each thread processes exactly one element
    if (tid < n) {
        output[tid] = input[tid] * input[tid] + 2.0f * input[tid] + 1.0f;
    }
}

// Kernel 2: Coarsened threading (MULTIPLE ELEMENTS PER THREAD)
__global__ void coarsenedKernel(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple elements using a stride pattern
    for (int i = tid; i < n; i += stride) {
        output[i] = input[i] * input[i] + 2.0f * input[i] + 1.0f;
    }
}

// Kernel 3: Student Exercise - Implement coarsened reduction
__global__ void studentCoarsenedReduction(float* input, float* output, int n) {
    // TODO: Implement a coarsened reduction where each thread handles multiple elements
    // HINT: Use a stride pattern similar to the coarsened kernel above
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int blockSize = blockDim.x;
    int gridSize = gridDim.x * blockSize;
    
    // FIX: Each thread should process multiple elements and contribute to a local sum
    float localSum = 0.0f;
    
    // FIX: Implement the stride loop to process multiple elements per thread
    for (int i = /* YOUR START INDEX */; i < n; i += /* YOUR STRIDE */) {
        localSum += input[i];
    }
    
    // Store local sum in shared memory for final reduction within block
    __shared__ float sdata[256];  // Assuming max 256 threads per block
    sdata[threadIdx.x] = localSum;
    __syncthreads();
    
    // Perform final reduction in shared memory
    for (int s = 1; s < blockSize; s *= 2) {
        if ((threadIdx.x % (2*s)) == 0 && (threadIdx.x + s) < blockSize) {
            sdata[threadIdx.x] += sdata[threadIdx.x + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (threadIdx.x == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Kernel 4: Student Exercise - Coarsened vector addition with different coarsening factors
__global__ void studentVariableCoarsening(float* A, float* B, float* C, int n, int coarsening_factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Implement variable coarsening where each thread processes 'coarsening_factor' elements
    // HINT: Calculate the range of elements each thread should process
    
    // FIX: Determine the starting and ending indices for this thread
    int start_idx = /* YOUR START INDEX */;
    int end_idx = /* YOUR END INDEX */;
    
    // Process multiple elements per thread
    for (int i = start_idx; i < end_idx && i < n; i++) {
        C[i] = A[i] + B[i];
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 100) * 0.1f;
    }
}

int main() {
    printf("=== Thread Coarsening Exercise ===\n");
    printf("Learn to use thread coarsening to improve kernel efficiency.\n\n");

    // Setup parameters
    const int N = 1000000;  // Large enough to see coarsening benefits
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output_fine, *h_output_coarse, *h_output_reduce;
    h_input = (float*)malloc(bytes);
    h_output_fine = (float*)malloc(bytes);
    h_output_coarse = (float*)malloc(bytes);
    h_output_reduce = (float*)malloc(N * sizeof(float));  // Temporary space for reductions
    
    // Initialize data
    initArray(h_input, N, 1.0f);
    
    // Allocate device memory
    float *d_input, *d_output_fine, *d_output_coarse, *d_output_reduce;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output_fine, bytes);
    cudaMalloc(&d_output_coarse, bytes);
    cudaMalloc(&d_output_reduce, N * sizeof(float));  // Temporary space for reductions
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Run fine-grained kernel
    printf("Running fine-grained kernel...\n");
    fineGrainedKernel<<<gridSize, blockSize>>>(d_input, d_output_fine, N);
    cudaDeviceSynchronize();
    
    // Run coarsened kernel
    printf("Running coarsened kernel...\n");
    coarsenedKernel<<<gridSize, blockSize>>>(d_input, d_output_coarse, N);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student coarsening exercises (complete the code first!)...\n");
    
    // Coarsened reduction exercise
    int reduce_grid_size = (N + blockSize - 1) / blockSize;
    studentCoarsenedReduction<<<reduce_grid_size, blockSize>>>(d_input, d_output_reduce, N);
    cudaDeviceSynchronize();
    
    // Variable coarsening exercise
    studentVariableCoarsening<<<gridSize, blockSize>>>(d_input, d_input, d_output_coarse, N, 4);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the coarsening implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_fine, d_output_fine, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_coarse, d_output_coarse, bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Input:      %.2f %.2f %.2f %.2f %.2f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    printf("Fine-grain: %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_fine[0], h_output_fine[1], h_output_fine[2], h_output_fine[3], h_output_fine[4]);
    printf("Coarsened:  %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_coarse[0], h_output_coarse[1], h_output_coarse[2], h_output_coarse[3], h_output_coarse[4]);
    
    // Cleanup
    free(h_input); free(h_output_fine); free(h_output_coarse); free(h_output_reduce);
    cudaFree(d_input); cudaFree(d_output_fine); cudaFree(d_output_coarse); cudaFree(d_output_reduce);
    
    printf("\nExercise completed! Notice how coarsening can reduce kernel launch overhead.\n");
    
    return 0;
}