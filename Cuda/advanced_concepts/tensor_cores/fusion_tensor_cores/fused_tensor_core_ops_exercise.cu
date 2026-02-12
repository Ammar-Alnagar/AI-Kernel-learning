/*
 * Fused Operations with Tensor Cores Exercise
 *
 * This exercise demonstrates how to fuse operations with Tensor Cores
 * to maximize performance and efficiency.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <wmma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// Kernel 1: Separate Operations (Tensor Core GEMM + Element-wise Activation)
__global__ void separateOps(half* A, half* B, float* temp, float* output, 
                          int M, int N, int K, float threshold) {
    // Tile using a 16x16x16 fragment
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Block and thread indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    // Bounds checking
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }
    
    // Allocate fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    
    // Initialize output to zero
    wmma::fill_fragment(fragC, 0.0f);
    
    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        // Load fragments
        wmma::load_matrix_sync(fragA, A + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(fragB, B + i * N + warpN * WMMA_N, N);
        
        // Matrix multiply-accumulate
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // Store the result to temporary buffer
    wmma::store_matrix_sync(temp + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
    
    // Apply activation function separately (this is inefficient)
    // In a real implementation, this would be a separate kernel launch
}

// Kernel 2: Fused Tensor Core Operations with Activation
__global__ void fusedTensorCoreOps(half* A, half* B, float* output, 
                                 int M, int N, int K, float threshold) {
    // Tile using a 16x16x16 fragment
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Block and thread indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    // Bounds checking
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }
    
    // Allocate fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    
    // Initialize output to zero
    wmma::fill_fragment(fragC, 0.0f);
    
    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        // Load fragments
        wmma::load_matrix_sync(fragA, A + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(fragB, B + i * N + warpN * WMMA_N, N);
        
        // Matrix multiply-accumulate
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // Apply activation function directly to the accumulator fragment
    // This is a simplified example - in practice, applying activation to WMMA fragments
    // requires element-wise access which is more complex
    for (int i = 0; i < fragC.num_elements; i++) {
        fragC.x[i] = fmaxf(0.0f, fragC.x[i]);  // ReLU activation
    }
    
    // Store the result with activation applied
    wmma::store_matrix_sync(output + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
}

// Kernel 3: Student Exercise - Implement fused GEMM with bias addition using Tensor Cores
__global__ void studentFusedGemmBias(half* A, half* B, float* bias, float* output, 
                                   int M, int N, int K) {
    // TODO: Implement fused GEMM + bias addition using Tensor Cores
    // HINT: Compute A*B using Tensor Cores, then add bias in the same kernel
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    // Bounds checking
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }
    
    // Allocate fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    
    // Initialize output to zero
    wmma::fill_fragment(fragC, 0.0f);
    
    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        // Load fragments
        wmma::load_matrix_sync(fragA, A + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(fragB, B + i * N + warpN * WMMA_N, N);
        
        // Matrix multiply-accumulate
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // FIX: Add bias to the accumulator fragment
    // HINT: You'll need to access individual elements of the accumulator fragment
    // and add the corresponding bias value
    
    // For now, just store the raw result
    wmma::store_matrix_sync(output + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
}

// Kernel 4: Student Exercise - Implement fused GEMM with Layer Normalization
__global__ void studentFusedGemmLayerNorm(half* A, half* B, float* gamma, float* beta, 
                                       float* output, int M, int N, int K, float eps) {
    // TODO: Implement fused GEMM with Layer Normalization
    // HINT: Compute A*B, then apply layer normalization (mean, variance, normalize, scale, shift)
    // This is a complex operation that requires multiple passes or approximation
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    // Bounds checking
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }
    
    // Allocate fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    
    // Initialize output to zero
    wmma::fill_fragment(fragC, 0.0f);
    
    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        // Load fragments
        wmma::load_matrix_sync(fragA, A + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(fragB, B + i * N + warpN * WMMA_N, N);
        
        // Matrix multiply-accumulate
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // FIX: Apply layer normalization to the result
    // This is quite complex and would typically require multiple kernel launches
    // or a simplified approximation for this exercise
    
    // For now, just store the raw result
    wmma::store_matrix_sync(output + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
}

// Kernel 5: Student Exercise - Implement fused operations with residual connection
__global__ void studentFusedWithResidual(half* A, half* B, half* residual, float* output, 
                                      int M, int N, int K, float alpha, float beta) {
    // TODO: Implement fused GEMM with residual connection: output = alpha * A*B + beta * residual
    // HINT: Compute A*B using Tensor Cores, then combine with residual in the same kernel
    
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    // Calculate indices
    int warpM = (blockIdx.y * blockDim.y + threadIdx.y) / warpSize;
    int warpN = (blockIdx.x * blockDim.x + threadIdx.x) / warpSize;
    int lane = threadIdx.x % warpSize;
    
    // Bounds checking
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }
    
    // Allocate fragments
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    
    // Initialize output to zero
    wmma::fill_fragment(fragC, 0.0f);
    
    // Loop over K dimension
    for (int i = 0; i < K; i += WMMA_K) {
        // Load fragments
        wmma::load_matrix_sync(fragA, A + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(fragB, B + i * N + warpN * WMMA_N, N);
        
        // Matrix multiply-accumulate
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // FIX: Scale the result by alpha and add beta * residual
    // HINT: You'll need to access individual elements of the accumulator fragment
    // and combine with residual values
    
    // For now, just store the raw result
    wmma::store_matrix_sync(output + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
}

// Utility function to initialize half matrix
void initHalfMatrix(half* mat, int rows, int cols, float start_val = 0.1f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = __float2half(start_val + (i % 100) * 0.01f);
    }
}

// Utility function to initialize float matrix
void initFloatMatrix(float* mat, int rows, int cols, float start_val = 0.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = start_val + (i % 100) * 0.01f;
    }
}

int main() {
    printf("=== Fused Operations with Tensor Cores Exercise ===\n");
    printf("Learn to fuse operations with Tensor Cores for maximum efficiency.\n\n");

    // Setup parameters
    const int M = 128, N = 128, K = 128;  // Must be multiples of 8, 16, or 32 depending on tensor core size
    size_t bytes_a = M * K * sizeof(half);
    size_t bytes_b = K * N * sizeof(half);
    size_t bytes_c = M * N * sizeof(float);
    size_t bytes_bias = N * sizeof(float);
    
    // Allocate host memory
    half *h_A, *h_B, *h_residual;
    float *h_temp, *h_output_separate, *h_output_fused, *h_output_bias, *h_output_layernorm, *h_output_residual;
    float *h_bias, *h_gamma, *h_beta;
    
    h_A = (half*)malloc(bytes_a);
    h_B = (half*)malloc(bytes_b);
    h_residual = (half*)malloc(bytes_c);
    h_temp = (float*)malloc(bytes_c);
    h_output_separate = (float*)malloc(bytes_c);
    h_output_fused = (float*)malloc(bytes_c);
    h_output_bias = (float*)malloc(bytes_c);
    h_output_layernorm = (float*)malloc(bytes_c);
    h_output_residual = (float*)malloc(bytes_c);
    h_bias = (float*)malloc(bytes_bias);
    h_gamma = (float*)malloc(bytes_bias);
    h_beta = (float*)malloc(bytes_bias);
    
    // Initialize matrices
    initHalfMatrix(h_A, M, K, 0.1f);
    initHalfMatrix(h_B, K, N, 0.2f);
    initHalfMatrix(h_residual, M, N, 0.05f);
    initFloatMatrix(h_bias, 1, N, 0.1f);
    initFloatMatrix(h_gamma, 1, N, 1.0f);
    initFloatMatrix(h_beta, 1, N, 0.0f);
    
    // Initialize output matrices to zero
    memset(h_output_separate, 0, bytes_c);
    memset(h_output_fused, 0, bytes_c);
    memset(h_output_bias, 0, bytes_c);
    memset(h_output_layernorm, 0, bytes_c);
    memset(h_output_residual, 0, bytes_c);
    
    // Allocate device memory
    half *d_A, *d_B, *d_residual;
    float *d_temp, *d_output_separate, *d_output_fused, *d_output_bias, *d_output_layernorm, *d_output_residual;
    float *d_bias, *d_gamma, *d_beta;
    
    cudaMalloc(&d_A, bytes_a);
    cudaMalloc(&d_B, bytes_b);
    cudaMalloc(&d_residual, bytes_c);
    cudaMalloc(&d_temp, bytes_c);
    cudaMalloc(&d_output_separate, bytes_c);
    cudaMalloc(&d_output_fused, bytes_c);
    cudaMalloc(&d_output_bias, bytes_c);
    cudaMalloc(&d_output_layernorm, bytes_c);
    cudaMalloc(&d_output_residual, bytes_c);
    cudaMalloc(&d_bias, bytes_bias);
    cudaMalloc(&d_gamma, bytes_bias);
    cudaMalloc(&d_beta, bytes_bias);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, h_residual, bytes_c, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bytes_bias, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gamma, h_gamma, bytes_bias, cudaMemcpyHostToDevice);
    cudaMemcpy(d_beta, h_beta, bytes_bias, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(32, 2);  // 32 threads per warp, 2 warps per block
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);  // 16x16 tiles
    
    // Run separate operations kernel
    printf("Running separate operations kernel...\n");
    separateOps<<<gridSize, blockSize>>>(d_A, d_B, d_temp, d_output_separate, M, N, K, 0.0f);
    cudaDeviceSynchronize();
    
    // Run fused operations kernel
    printf("Running fused operations kernel...\n");
    fusedTensorCoreOps<<<gridSize, blockSize>>>(d_A, d_B, d_output_fused, M, N, K, 0.0f);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student fused Tensor Core exercises (complete the code first!)...\n");
    
    // Fused GEMM + bias exercise
    studentFusedGemmBias<<<gridSize, blockSize>>>(d_A, d_B, d_bias, d_output_bias, M, N, K);
    cudaDeviceSynchronize();
    
    // Fused GEMM + LayerNorm exercise
    studentFusedGemmLayerNorm<<<gridSize, blockSize>>>(d_A, d_B, d_gamma, d_beta, d_output_layernorm, M, N, K, 1e-5f);
    cudaDeviceSynchronize();
    
    // Fused operations with residual exercise
    float alpha = 1.0f, beta = 1.0f;
    studentFusedWithResidual<<<gridSize, blockSize>>>(d_A, d_B, d_residual, d_output_residual, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the fused Tensor Core implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_separate, d_output_separate, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused, d_output_fused, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_bias, d_output_bias, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_layernorm, d_output_layernorm, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_residual, d_output_residual, bytes_c, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Separate ops: %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_separate[0], h_output_separate[1], h_output_separate[2], h_output_separate[3], h_output_separate[4]);
    printf("Fused ops:    %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_fused[0], h_output_fused[1], h_output_fused[2], h_output_fused[3], h_output_fused[4]);
    
    // Cleanup
    free(h_A); free(h_B); free(h_residual);
    free(h_temp); free(h_output_separate); free(h_output_fused);
    free(h_output_bias); free(h_output_layernorm); free(h_output_residual);
    free(h_bias); free(h_gamma); free(h_beta);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_residual);
    cudaFree(d_temp); cudaFree(d_output_separate); cudaFree(d_output_fused);
    cudaFree(d_output_bias); cudaFree(d_output_layernorm); cudaFree(d_output_residual);
    cudaFree(d_bias); cudaFree(d_gamma); cudaFree(d_beta);
    
    printf("\nExercise completed! Notice how fusing operations with Tensor Cores improves efficiency.\n");
    
    return 0;
}