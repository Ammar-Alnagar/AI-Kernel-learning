/*
 * Online Reductions Exercise
 *
 * This exercise demonstrates how to implement online reductions that process
 * data in a single pass, useful for streaming data or when memory is limited.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Standard Reduction (Two-Pass)
__global__ void standardReduction(float* input, float* output, int n) {
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

// Kernel 2: Online Reduction (Single-Pass with Count and Mean)
__global__ void onlineReduction(float* input, float* means, float* counts, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize shared memory with current value or 0
    float current_val = (i < n) ? input[i] : 0.0f;
    sdata[tid] = current_val;
    __syncthreads();
    
    // Perform online mean calculation in shared memory
    // This simulates processing streaming data where we update statistics incrementally
    float local_mean = (i < n) ? current_val : 0.0f;
    float local_count = (i < n) ? 1.0f : 0.0f;
    
    // Perform reduction to compute mean incrementally
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            float incoming_mean = sdata[tid + s];
            float incoming_count = 1.0f; // Simplified - in real online scenario, this would come from previous calculations
            
            // Online mean update: new_mean = (old_mean * old_count + new_val * 1) / (old_count + 1)
            float merged_count = local_count + incoming_count;
            if (merged_count > 0) {
                local_mean = (local_mean * local_count + incoming_mean * incoming_count) / merged_count;
                local_count = merged_count;
            }
        }
        __syncthreads();
    }
    
    // Write results for this block to global memory
    if (tid == 0) {
        means[blockIdx.x] = local_mean;
        counts[blockIdx.x] = local_count;
    }
}

// Kernel 3: Student Exercise - Implement online variance reduction
__global__ void studentOnlineVariance(float* input, float* means, float* vars, float* counts, int n) {
    // TODO: Implement online variance calculation using Welford's algorithm
    // HINT: Maintain running count, mean, and sum of squares for incremental variance calculation
    // Formula: variance = (sum_of_squares - (sum^2)/count) / count
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Initialize local statistics for this thread
    float local_count = 0.0f;
    float local_mean = 0.0f;
    float local_m2 = 0.0f;  // Sum of squares of differences from mean
    
    if (i < n) {
        float value = input[i];
        
        // FIX: Implement Welford's online algorithm for variance
        // Step 1: Initialize with first value
        local_count = 1.0f;
        local_mean = value;
        local_m2 = 0.0f;  // Variance of single value is 0
        
        // In a real streaming scenario, you would update these values incrementally
        // For this exercise, we'll simulate the online update process
    }
    
    // Store in shared memory for block-level combination
    // We'll use multiple shared memory arrays to store the statistics
    __shared__ float shared_counts[256];
    __shared__ float shared_means[256];
    __shared__ float shared_m2s[256];
    
    shared_counts[tid] = local_count;
    shared_means[tid] = local_mean;
    shared_m2s[tid] = local_m2;
    __syncthreads();
    
    // Perform online combination of statistics across threads in block
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            // FIX: Combine statistics from two groups using online algorithm
            float count_a = shared_counts[tid];
            float mean_a = shared_means[tid];
            float m2_a = shared_m2s[tid];
            
            float count_b = shared_counts[tid + s];
            float mean_b = shared_means[tid + s];
            float m2_b = shared_m2s[tid + s];
            
            // Combine counts
            float combined_count = count_a + count_b;
            
            if (combined_count > 0) {
                // Combine means
                float delta = mean_b - mean_a;
                float combined_mean = mean_a + delta * count_b / combined_count;
                
                // Combine M2 (sum of squares)
                float combined_m2 = m2_a + m2_b + delta * delta * count_a * count_b / combined_count;
                
                // Store combined statistics
                shared_counts[tid] = combined_count;
                shared_means[tid] = combined_mean;
                shared_m2s[tid] = combined_m2;
            }
        }
        __syncthreads();
    }
    
    // Write final statistics for this block to global memory
    if (tid == 0) {
        means[blockIdx.x] = shared_means[0];
        counts[blockIdx.x] = shared_counts[0];
        vars[blockIdx.x] = (shared_counts[0] > 1) ? shared_m2s[0] / shared_counts[0] : 0.0f;  // Population variance
    }
}

// Kernel 4: Student Exercise - Implement online min/max reduction
__global__ void studentOnlineMinMax(float* input, float* mins, float* maxs, int n) {
    // TODO: Implement online min/max calculation
    // HINT: Track running minimum and maximum values as new data arrives
    
    extern __shared__ float sdata_min[];
    extern __shared__ float sdata_max[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIX: Initialize min/max values appropriately
    float local_min = (i < n) ? input[i] : FLT_MAX;  // Start with largest possible value
    float local_max = (i < n) ? input[i] : -FLT_MAX; // Start with smallest possible value
    
    // Store in shared memory for reduction
    sdata_min[tid] = local_min;
    sdata_max[tid] = local_max;
    __syncthreads();
    
    // Perform reduction to find min/max across threads in block
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            // FIX: Update min and max values by comparing with adjacent thread's values
            sdata_min[tid] = fminf(sdata_min[tid], sdata_min[tid + s]);
            sdata_max[tid] = fmaxf(sdata_max[tid], sdata_max[tid + s]);
        }
        __syncthreads();
    }
    
    // Write results for this block to global memory
    if (tid == 0) {
        mins[blockIdx.x] = sdata_min[0];
        maxs[blockIdx.x] = sdata_max[0];
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 100) * 0.1f;
    }
}

int main() {
    printf("=== Online Reductions Exercise ===\n");
    printf("Learn to implement online reduction algorithms for streaming data.\n\n");

    // Setup parameters
    const int N = 1024 * 16;  // Multiple of block size
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    size_t bytes = N * sizeof(float);
    size_t output_bytes = gridSize * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output_standard, *h_means, *h_vars, *h_counts, *h_mins, *h_maxs;
    h_input = (float*)malloc(bytes);
    h_output_standard = (float*)malloc(output_bytes);
    h_means = (float*)malloc(output_bytes);
    h_vars = (float*)malloc(output_bytes);
    h_counts = (float*)malloc(output_bytes);
    h_mins = (float*)malloc(output_bytes);
    h_maxs = (float*)malloc(output_bytes);
    
    // Initialize data
    initArray(h_input, N, 1.0f);
    
    // Allocate device memory
    float *d_input, *d_output_standard, *d_means, *d_vars, *d_counts, *d_mins, *d_maxs;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output_standard, output_bytes);
    cudaMalloc(&d_means, output_bytes);
    cudaMalloc(&d_vars, output_bytes);
    cudaMalloc(&d_counts, output_bytes);
    cudaMalloc(&d_mins, output_bytes);
    cudaMalloc(&d_maxs, output_bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Define shared memory size
    size_t shared_mem_size = blockSize * sizeof(float);
    
    // Run standard reduction kernel
    printf("Running standard reduction kernel...\n");
    standardReduction<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_output_standard, N);
    cudaDeviceSynchronize();
    
    // Run online reduction kernel
    printf("Running online reduction kernel...\n");
    onlineReduction<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_means, d_counts, N);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student online reduction exercises (complete the code first!)...\n");
    
    // Online variance exercise
    studentOnlineVariance<<<gridSize, blockSize, shared_mem_size * 3>>>(d_input, d_means, d_vars, d_counts, N);
    cudaDeviceSynchronize();
    
    // Online min/max exercise
    studentOnlineMinMax<<<gridSize, blockSize, shared_mem_size * 2>>>(d_input, d_mins, d_maxs, N);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the online reduction implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_standard, d_output_standard, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_means, d_means, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vars, d_vars, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_counts, d_counts, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_mins, d_mins, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_maxs, d_maxs, output_bytes, cudaMemcpyDeviceToHost);
    
    // Calculate total from block results
    float total_standard = 0.0f;
    for (int i = 0; i < gridSize; i++) {
        total_standard += h_output_standard[i];
    }
    
    // Print sample results
    printf("\nSample results:\n");
    printf("Standard reduction total: %.2f\n", total_standard);
    printf("Online mean (first block): %.2f\n", h_means[0]);
    printf("Online count (first block): %.2f\n", h_counts[0]);
    printf("Online variance (first block): %.2f\n", h_vars[0]);
    printf("Online min (first block): %.2f\n", h_mins[0]);
    printf("Online max (first block): %.2f\n", h_maxs[0]);
    
    // Cleanup
    free(h_input); free(h_output_standard); free(h_means); 
    free(h_vars); free(h_counts); free(h_mins); free(h_maxs);
    cudaFree(d_input); cudaFree(d_output_standard); cudaFree(d_means);
    cudaFree(d_vars); cudaFree(d_counts); cudaFree(d_mins); cudaFree(d_maxs);
    
    printf("\nExercise completed! Notice how online reductions process data in single passes.\n");
    
    return 0;
}