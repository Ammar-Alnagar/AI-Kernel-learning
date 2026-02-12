/*
 * Layer Normalization Exercise
 *
 * This exercise demonstrates how to implement efficient Layer Normalization,
 * a key component in transformer architectures.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Kernel 1: Naive Layer Normalization (INEFFICIENT)
__global__ void naiveLayerNorm(float* input, float* output, float* gamma, float* beta,
                              int batch_size, int hidden_size, float eps) {
    int batch_idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Compute mean
        float sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            sum += input[batch_idx * hidden_size + i];
        }
        float mean = sum / hidden_size;
        
        // Compute variance
        float var_sum = 0.0f;
        for (int i = 0; i < hidden_size; i++) {
            float diff = input[batch_idx * hidden_size + i] - mean;
            var_sum += diff * diff;
        }
        float variance = var_sum / hidden_size;
        
        // Normalize and apply affine transformation
        for (int i = 0; i < hidden_size; i++) {
            float normalized = (input[batch_idx * hidden_size + i] - mean) / 
                              sqrtf(variance + eps);
            output[batch_idx * hidden_size + i] = 
                gamma[i] * normalized + beta[i];
        }
    }
}

// Kernel 2: Optimized Layer Normalization (Using Shared Memory)
__global__ void optimizedLayerNorm(float* input, float* output, float* gamma, float* beta,
                                  int batch_size, int hidden_size, float eps) {
    extern __shared__ float sdata[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // Each thread processes multiple elements if hidden_size > num_threads
        int elements_per_thread = (hidden_size + blockDim.x - 1) / blockDim.x;
        int start_idx = tid * elements_per_thread;
        int end_idx = min(start_idx + elements_per_thread, hidden_size);
        
        // Compute local sum for this thread's elements
        float local_sum = 0.0f;
        for (int i = start_idx; i < end_idx; i++) {
            local_sum += input[batch_idx * hidden_size + i];
        }
        
        // Store in shared memory for reduction
        sdata[tid] = local_sum;
        __syncthreads();
        
        // Perform reduction to get total sum
        for (int s = 1; s < blockDim.x; s *= 2) {
            if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        float mean = sdata[0] / hidden_size;
        __syncthreads();
        
        // Compute local variance sum
        float local_var_sum = 0.0f;
        for (int i = start_idx; i < end_idx; i++) {
            float diff = input[batch_idx * hidden_size + i] - mean;
            local_var_sum += diff * diff;
        }
        
        // Store in shared memory for variance reduction
        sdata[tid] = local_var_sum;
        __syncthreads();
        
        // Perform reduction to get total variance
        for (int s = 1; s < blockDim.x; s *= 2) {
            if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        float variance = sdata[0] / hidden_size;
        float inv_std = rsqrtf(variance + eps);  // 1/sqrt(x) = rsqrt(x)
        
        // Apply normalization and affine transformation
        for (int i = start_idx; i < end_idx; i++) {
            float normalized = (input[batch_idx * hidden_size + i] - mean) * inv_std;
            output[batch_idx * hidden_size + i] = 
                gamma[i] * normalized + beta[i];
        }
    }
}

// Kernel 3: Student Exercise - Implement online Layer Normalization
__global__ void studentOnlineLayerNorm(float* input, float* output, float* gamma, float* beta,
                                      int batch_size, int hidden_size, float eps) {
    // TODO: Implement online Layer Normalization that computes mean and variance
    // in a single pass to reduce memory accesses
    // HINT: Use Welford's online algorithm for numerically stable variance calculation
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // FIX: Implement online mean/variance calculation in a single pass
        // Use Welford's algorithm: https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Welford's_online_algorithm
        
        // Placeholder for now
        float mean = 0.0f;
        float variance = 0.0f;
        
        // FIX: Calculate mean and variance in a single pass
        // for (int i = 0; i < hidden_size; i++) {
        //     // Apply Welford's algorithm here
        // }
        
        // Then apply normalization
        int elements_per_thread = (hidden_size + blockDim.x - 1) / blockDim.x;
        int start_idx = tid * elements_per_thread;
        int end_idx = min(start_idx + elements_per_thread, hidden_size);
        
        for (int i = start_idx; i < end_idx; i++) {
            float normalized = (input[batch_idx * hidden_size + i] - mean) / 
                              sqrtf(variance + eps);
            output[batch_idx * hidden_size + i] = 
                gamma[i] * normalized + beta[i];
        }
    }
}

// Kernel 4: Student Exercise - Implement fused Layer Normalization with residual connection
__global__ void studentFusedLayerNormResidual(float* input, float* residual, float* output, 
                                             float* gamma, float* beta, int batch_size, 
                                             int hidden_size, float eps) {
    // TODO: Implement fused Layer Normalization that adds residual connection
    // before applying normalization: output = LayerNorm(input + residual)
    // HINT: Add input and residual first, then apply normalization
    
    extern __shared__ float sdata[];
    
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    
    if (batch_idx < batch_size) {
        // FIX: Add residual connection first
        int elements_per_thread = (hidden_size + blockDim.x - 1) / blockDim.x;
        int start_idx = tid * elements_per_thread;
        int end_idx = min(start_idx + elements_per_thread, hidden_size);
        
        // FIX: Add input and residual to create the input for layer norm
        float local_input[64];  // Assuming max elements per thread
        for (int i = start_idx, j = 0; i < end_idx; i++, j++) {
            local_input[j] = /* INPUT + RESIDUAL */;
        }
        
        // FIX: Compute mean of the combined input
        float local_sum = 0.0f;
        for (int i = start_idx, j = 0; i < end_idx; i++, j++) {
            local_sum += local_input[j];
        }
        sdata[tid] = local_sum;
        __syncthreads();
        
        // Perform reduction to get total sum
        for (int s = 1; s < blockDim.x; s *= 2) {
            if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        float mean = sdata[0] / hidden_size;
        __syncthreads();
        
        // FIX: Compute variance using the combined input
        float local_var_sum = 0.0f;
        for (int i = start_idx, j = 0; i < end_idx; i++, j++) {
            float diff = local_input[j] - mean;
            local_var_sum += diff * diff;
        }
        sdata[tid] = local_var_sum;
        __syncthreads();
        
        // Perform reduction to get total variance
        for (int s = 1; s < blockDim.x; s *= 2) {
            if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        float variance = sdata[0] / hidden_size;
        float inv_std = rsqrtf(variance + eps);
        
        // Apply normalization and affine transformation
        for (int i = start_idx, j = 0; i < end_idx; i++, j++) {
            float normalized = (local_input[j] - mean) * inv_std;
            output[batch_idx * hidden_size + i] = 
                gamma[i] * normalized + beta[i];
        }
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int rows, int cols, float start_val = 0.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = start_val + (i % 100) * 0.01f - 0.5f;
    }
}

// Utility function to initialize vector
void initVector(float* vec, int size, float start_val = 1.0f) {
    for (int i = 0; i < size; i++) {
        vec[i] = start_val + (i % 10) * 0.1f;
    }
}

int main() {
    printf("=== Layer Normalization Exercise ===\n");
    printf("Learn to implement efficient Layer Normalization kernels.\n\n");

    // Setup parameters
    const int BATCH_SIZE = 16;
    const int HIDDEN_SIZE = 128;
    const float EPSILON = 1e-5f;
    
    size_t input_bytes = BATCH_SIZE * HIDDEN_SIZE * sizeof(float);
    size_t param_bytes = HIDDEN_SIZE * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_gamma, *h_beta;
    float *h_output_naive, *h_output_opt, *h_output_online, *h_output_fused;
    float *h_residual;
    
    h_input = (float*)malloc(input_bytes);
    h_gamma = (float*)malloc(param_bytes);
    h_beta = (float*)malloc(param_bytes);
    h_output_naive = (float*)malloc(input_bytes);
    h_output_opt = (float*)malloc(input_bytes);
    h_output_online = (float*)malloc(input_bytes);
    h_output_fused = (float*)malloc(input_bytes);
    h_residual = (float*)malloc(input_bytes);
    
    // Initialize data
    initMatrix(h_input, BATCH_SIZE, HIDDEN_SIZE, 0.1f);
    initVector(h_gamma, HIDDEN_SIZE, 1.0f);
    initVector(h_beta, HIDDEN_SIZE, 0.0f);
    initMatrix(h_residual, BATCH_SIZE, HIDDEN_SIZE, 0.05f);
    
    // Allocate device memory
    float *d_input, *d_gamma, *d_beta;
    float *d_output_naive, *d_output_opt, *d_output_online, *d_output_fused;
    float *d_residual;
    
    cudaMalloc(&d_input, input_bytes);
    cudaMalloc(&d_gamma, param_bytes);
    cudaMalloc(&d_beta, param_bytes);
    cudaMalloc(&d_output_naive, input_bytes);
    cudaMalloc(&d_output_opt, input_bytes);
    cudaMalloc(&d_output_online, input_bytes);
    cudaMalloc(&d_output_fused, input_bytes);
    cudaMalloc(&d_residual, input_bytes);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, input_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, param_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, h_residual, input_bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = min(HIDDEN_SIZE, 256);
    int gridSize = BATCH_SIZE;
    size_t shared_mem_size = blockSize * sizeof(float);
    
    // Run naive Layer Norm kernel
    printf("Running naive Layer Normalization kernel...\n");
    naiveLayerNorm<<<gridSize, blockSize>>>(d_input, d_output_naive, d_gamma, d_beta,
                                           BATCH_SIZE, HIDDEN_SIZE, EPSILON);
    cudaDeviceSynchronize();
    
    // Run optimized Layer Norm kernel
    printf("Running optimized Layer Normalization kernel...\n");
    optimizedLayerNorm<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_output_opt, 
                                                                d_gamma, d_beta, 
                                                                BATCH_SIZE, HIDDEN_SIZE, EPSILON);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student Layer Normalization exercises (complete the code first!)...\n");
    
    // Online Layer Norm exercise
    studentOnlineLayerNorm<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_output_online, 
                                                                   d_gamma, d_beta, 
                                                                   BATCH_SIZE, HIDDEN_SIZE, EPSILON);
    cudaDeviceSynchronize();
    
    // Fused Layer Norm with residual exercise
    studentFusedLayerNormResidual<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_residual, 
                                                                          d_output_fused, 
                                                                          d_gamma, d_beta, 
                                                                          BATCH_SIZE, HIDDEN_SIZE, 
                                                                          EPSILON);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the Layer Normalization implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_naive, d_output_naive, input_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_opt, d_output_opt, input_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_online, d_output_online, input_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused, d_output_fused, input_bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements of first batch):\n");
    printf("Input:     %.3f %.3f %.3f %.3f %.3f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    printf("Naive LN:  %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_naive[0], h_output_naive[1], h_output_naive[2], h_output_naive[3], h_output_naive[4]);
    printf("Opt LN:    %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_opt[0], h_output_opt[1], h_output_opt[2], h_output_opt[3], h_output_opt[4]);
    
    // Cleanup
    free(h_input); free(h_gamma); free(h_beta);
    free(h_output_naive); free(h_output_opt); free(h_output_online); free(h_output_fused);
    free(h_residual);
    cudaFree(d_input); cudaFree(d_gamma); cudaFree(d_beta);
    cudaFree(d_output_naive); cudaFree(d_output_opt); cudaFree(d_output_online); cudaFree(d_output_fused);
    cudaFree(d_residual);
    
    printf("\nExercise completed! Notice how optimized implementations improve Layer Normalization performance.\n");
    
    return 0;
}