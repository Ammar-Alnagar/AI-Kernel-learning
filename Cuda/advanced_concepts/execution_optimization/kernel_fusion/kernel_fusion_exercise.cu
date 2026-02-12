/*
 * Kernel Fusion Exercise
 *
 * This exercise demonstrates how to fuse multiple operations into a single kernel
 * to reduce memory traffic and kernel launch overhead.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Kernel 1: Separate kernels (UNFUSED - LESS EFFICIENT)
__global__ void kernel1_vecAdd(float* A, float* B, float* temp, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        temp[tid] = A[tid] + B[tid];  // First operation
    }
}

__global__ void kernel2_applyFunc(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = sqrtf(fabsf(input[tid])) * 2.0f + 1.0f;  // Second operation
    }
}

// Kernel 2: Fused kernel (MORE EFFICIENT)
__global__ void fusedKernel(float* A, float* B, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        float temp = A[tid] + B[tid];              // First operation
        output[tid] = sqrtf(fabsf(temp)) * 2.0f + 1.0f;  // Second operation
    }
}

// Kernel 3: Student Exercise - Fuse matrix multiplication and activation
__global__ void studentFusedMatmulActivation(float* A, float* B, float* C, float* bias, int width) {
    // TODO: Fuse matrix multiplication with activation function (ReLU)
    // HINT: Compute matrix multiplication result and apply ReLU in the same kernel
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        // FIX: Compute matrix multiplication: C[row][col] = sum(A[row][k] * B[k][col])
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        
        // FIX: Add bias and apply ReLU activation in the same step
        float biased_result = /* ADD BIAS HERE */;
        float activated_result = /* APPLY ReLU HERE */;
        
        C[row * width + col] = activated_result;
    }
}

// Kernel 4: Student Exercise - Fuse element-wise operations
__global__ void studentFusedElementwise(float* A, float* B, float* C, float* D, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // TODO: Fuse multiple element-wise operations into a single kernel
        // Instead of: temp1 = A + B, temp2 = C * D, output = temp1 * temp2
        // Do it all in one step
        
        // FIX: Combine all operations in a single computation
        float result = /* YOUR FUSED COMPUTATION */;
        
        output[tid] = result;
    }
}

// Kernel 5: Student Exercise - Implement GELU + Add fusion
__global__ void studentFusedGeluAdd(float* input, float* residual, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // TODO: Fuse GELU activation with residual connection
        // Formula: output = GELU(input) + residual
        
        float x = input[tid];
        
        // FIX: Compute GELU(x) = x * 0.5 * (1.0 + tanh(sqrt(2.0/M_PI) * (x + 0.044715 * x^3)))
        float gelu_result = /* YOUR GELU COMPUTATION */;
        
        // FIX: Add residual connection
        output[tid] = /* GELU RESULT + RESIDUAL */;
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 20 - 10) * 0.1f;  // Mix of positive/negative values
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int width, float start_val = 1.0f) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = start_val + (i % 20 - 10) * 0.01f;
    }
}

int main() {
    printf("=== Kernel Fusion Exercise ===\n");
    printf("Learn to fuse multiple operations into single kernels for better performance.\n\n");

    // Setup parameters for vector operations
    const int N = 1024 * 16;
    size_t bytes = N * sizeof(float);
    
    // Setup parameters for matrix operations
    const int WIDTH = 64;
    const int MAT_SIZE = WIDTH * WIDTH;
    size_t mat_bytes = MAT_SIZE * sizeof(float);
    
    // Allocate host memory for vector operations
    float *h_A, *h_B, *h_temp, *h_output_unfused, *h_output_fused;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_temp = (float*)malloc(bytes);
    h_output_unfused = (float*)malloc(bytes);
    h_output_fused = (float*)malloc(bytes);
    
    // Initialize vector data
    initArray(h_A, N, 1.0f);
    initArray(h_B, N, 2.0f);
    
    // Allocate device memory for vector operations
    float *d_A, *d_B, *d_temp, *d_output_unfused, *d_output_fused;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_temp, bytes);
    cudaMalloc(&d_output_unfused, bytes);
    cudaMalloc(&d_output_fused, bytes);
    
    // Copy vector data to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    int mat_gridSize = (WIDTH + 15) / 16;  // For 16x16 blocks
    dim3 mat_blockSize(16, 16);
    dim3 mat_gridSize_2D(mat_gridSize, mat_gridSize);
    
    // Run unfused kernels
    printf("Running unfused kernels...\n");
    kernel1_vecAdd<<<gridSize, blockSize>>>(d_A, d_B, d_temp, N);
    kernel2_applyFunc<<<gridSize, blockSize>>>(d_temp, d_output_unfused, N);
    cudaDeviceSynchronize();
    
    // Run fused kernel
    printf("Running fused kernel...\n");
    fusedKernel<<<gridSize, blockSize>>>(d_A, d_B, d_output_fused, N);
    cudaDeviceSynchronize();
    
    // Setup for matrix operations
    float *h_matA, *h_matB, *h_bias, *h_output_fused_mat;
    h_matA = (float*)malloc(mat_bytes);
    h_matB = (float*)malloc(mat_bytes);
    h_bias = (float*)malloc(mat_bytes);
    h_output_fused_mat = (float*)malloc(mat_bytes);
    
    // Initialize matrix data
    initMatrix(h_matA, WIDTH, 1.0f);
    initMatrix(h_matB, WIDTH, 2.0f);
    initArray(h_bias, MAT_SIZE, 0.1f);
    
    // Allocate device memory for matrix operations
    float *d_matA, *d_matB, *d_bias, *d_output_fused_mat;
    cudaMalloc(&d_matA, mat_bytes);
    cudaMalloc(&d_matB, mat_bytes);
    cudaMalloc(&d_bias, mat_bytes);
    cudaMalloc(&d_output_fused_mat, mat_bytes);
    
    // Copy matrix data to device
    cudaMemcpy(d_matA, h_matA, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_matB, h_matB, mat_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, mat_bytes, cudaMemcpyHostToDevice);
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student fusion exercises (complete the code first!)...\n");
    
    // Fused matmul + activation exercise
    studentFusedMatmulActivation<<<mat_gridSize_2D, mat_blockSize>>>(d_matA, d_matB, d_output_fused_mat, d_bias, WIDTH);
    cudaDeviceSynchronize();
    
    // Fused elementwise exercise
    studentFusedElementwise<<<gridSize, blockSize>>>(d_A, d_B, d_A, d_B, d_output_fused, N);
    cudaDeviceSynchronize();
    
    // Fused GELU + Add exercise
    studentFusedGeluAdd<<<gridSize, blockSize>>>(d_A, d_B, d_output_fused, N);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the fusion implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_unfused, d_output_unfused, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused, d_output_fused, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused_mat, d_output_fused_mat, mat_bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Input A:      %.2f %.2f %.2f %.2f %.2f\n", 
           h_A[0], h_A[1], h_A[2], h_A[3], h_A[4]);
    printf("Unfused:      %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_unfused[0], h_output_unfused[1], h_output_unfused[2], h_output_unfused[3], h_output_unfused[4]);
    printf("Fused:        %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_fused[0], h_output_fused[1], h_output_fused[2], h_output_fused[3], h_output_fused[4]);
    
    // Cleanup
    free(h_A); free(h_B); free(h_temp); free(h_output_unfused); free(h_output_fused);
    free(h_matA); free(h_matB); free(h_bias); free(h_output_fused_mat);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_temp); cudaFree(d_output_unfused); cudaFree(d_output_fused);
    cudaFree(d_matA); cudaFree(d_matB); cudaFree(d_bias); cudaFree(d_output_fused_mat);
    
    printf("\nExercise completed! Notice how kernel fusion reduces memory traffic.\n");
    
    return 0;
}