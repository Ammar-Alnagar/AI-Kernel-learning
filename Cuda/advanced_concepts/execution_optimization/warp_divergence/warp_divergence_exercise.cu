/*
 * Warp Divergence Exercise
 *
 * This exercise demonstrates how warp divergence affects GPU performance
 * and how to minimize its impact.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: With Warp Divergence (INEFFICIENT)
__global__ void divergentKernel(float* input, float* output, int n, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // DIVERGENT PATH: Threads in the same warp may take different branches
        if (input[tid] > threshold) {
            // Expensive computation path
            float temp = input[tid];
            for (int i = 0; i < 100; i++) {
                temp = temp * temp + 1.0f;
            }
            output[tid] = temp;
        } else {
            // Cheap computation path
            output[tid] = input[tid] * 2.0f;
        }
    }
}

// Kernel 2: Without Warp Divergence (EFFICIENT)
__global__ void convergedKernel(float* input, float* output, int n, float threshold) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float val = input[tid];
        
        // NON-DIVERGENT: Same computation for all threads, with conditional assignment
        float expensive_result = val;
        for (int i = 0; i < 100; i++) {
            expensive_result = expensive_result * expensive_result + 1.0f;
        }
        float cheap_result = val * 2.0f;
        
        // Conditional assignment instead of conditional execution
        output[tid] = (val > threshold) ? expensive_result : cheap_result;
    }
}

// Kernel 3: Student Exercise - Fix warp divergence in sorting network
__global__ void studentSortingNetwork(float* data, int n, int power_of_2) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int local_tid = threadIdx.x;
    
    // TODO: Fix warp divergence in this bitonic sort implementation
    // Current implementation has divergent comparisons within warps
    
    if (tid < n) {
        // FIX: Implement a sorting network that minimizes warp divergence
        // HINT: Consider how to organize comparisons to keep threads in a warp synchronized
        
        // Current problematic approach:
        /*
        for (int k = 2; k <= power_of_2; k <<= 1) {  // Double k each iteration
            for (int j = k >> 1; j > 0; j >>= 1) {   // Halve j each iteration
                int ixj = tid ^ j;  // XOR to find pair
                
                if (ixj > tid) {  // Prevent double work
                    if (((tid & k) == 0) ? (data[tid] > data[ixj]) : (data[tid] < data[ixj])) {
                        // Swap elements
                        float temp = data[tid];
                        data[tid] = data[ixj];
                        data[ixj] = temp;
                    }
                }
                __syncthreads();  // Synchronize after each step
            }
        }
        */
        
        // YOUR FIXED IMPLEMENTATION HERE:
        /* IMPLEMENT DIVERGENCE-FREE SORTING NETWORK */;
    }
}

// Kernel 4: Student Exercise - Minimize divergence in conditional processing
__global__ void studentConditionalProcessing(float* input, float* output, int n, int* categories) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        float val = input[tid];
        int category = categories[tid];
        
        // TODO: Fix this divergent conditional structure
        // Current implementation causes threads in the same warp to follow different paths
        
        // Current divergent approach (INEFFICIENT):
        /*
        if (category == 0) {
            // Process as type A
            output[tid] = val * val + 2.0f * val + 1.0f;
        } else if (category == 1) {
            // Process as type B
            output[tid] = sqrtf(fabsf(val)) * 10.0f;
        } else if (category == 2) {
            // Process as type C
            output[tid] = expf(-val) * 100.0f;
        } else {
            // Process as type D
            output[tid] = logf(fmaxf(val, 1e-6f)) * 5.0f;
        }
        */
        
        // YOUR EFFICIENT NON-DIVERGENT IMPLEMENTATION:
        /* CALCULATE ALL POSSIBLE RESULTS AND SELECT THE APPROPRIATE ONE */;
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 100) * 0.1f;
    }
}

// Utility function to initialize categories randomly
void initCategories(int* cats, int n) {
    for (int i = 0; i < n; i++) {
        cats[i] = i % 4;  // Distribute among 4 categories
    }
}

int main() {
    printf("=== Warp Divergence Exercise ===\n");
    printf("Learn to identify and minimize warp divergence in CUDA kernels.\n\n");

    // Setup parameters
    const int N = 1024;
    size_t bytes = N * sizeof(float);
    size_t int_bytes = N * sizeof(int);
    
    // Allocate host memory
    float *h_input, *h_output_div, *h_output_conv;
    int *h_categories;
    h_input = (float*)malloc(bytes);
    h_output_div = (float*)malloc(bytes);
    h_output_conv = (float*)malloc(bytes);
    h_categories = (int*)malloc(int_bytes);
    
    // Initialize data
    initArray(h_input, N, 0.0f);
    initCategories(h_categories, N);
    
    // Allocate device memory
    float *d_input, *d_output_div, *d_output_conv;
    int *d_categories;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output_div, bytes);
    cudaMalloc(&d_output_conv, bytes);
    cudaMalloc(&d_categories, int_bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_categories, h_categories, int_bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    float threshold = 50.0f;
    
    // Run divergent kernel
    printf("Running divergent kernel...\n");
    divergentKernel<<<gridSize, blockSize>>>(d_input, d_output_div, N, threshold);
    cudaDeviceSynchronize();
    
    // Run converged kernel
    printf("Running converged kernel...\n");
    convergedKernel<<<gridSize, blockSize>>>(d_input, d_output_conv, N, threshold);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student warp divergence exercises (complete the code first!)...\n");
    
    // Sorting network exercise
    studentSortingNetwork<<<gridSize, blockSize>>>(d_input, d_output_div, N, 256);
    cudaDeviceSynchronize();
    
    // Conditional processing exercise
    studentConditionalProcessing<<<gridSize, blockSize>>>(d_input, d_output_conv, N, d_categories);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the divergence-minimizing implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_div, d_output_div, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_conv, d_output_conv, bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Input:     %.2f %.2f %.2f %.2f %.2f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    printf("Divergent: %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_div[0], h_output_div[1], h_output_div[2], h_output_div[3], h_output_div[4]);
    printf("Converged: %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_conv[0], h_output_conv[1], h_output_conv[2], h_output_conv[3], h_output_conv[4]);
    
    // Cleanup
    free(h_input); free(h_output_div); free(h_output_conv); free(h_categories);
    cudaFree(d_input); cudaFree(d_output_div); cudaFree(d_output_conv); cudaFree(d_categories);
    
    printf("\nExercise completed! Notice how minimizing divergence can improve performance.\n");
    
    return 0;
}