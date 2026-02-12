/*
 * Tensor Cores Exercise
 *
 * This exercise demonstrates how to use Tensor Cores for high-performance matrix multiplication.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <wmma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// Kernel 1: Regular GEMM (without Tensor Cores)
__global__ void regularGemm(half* a, half* b, float* c, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += __half2float(a[row * K + k]) * __half2float(b[k * N + col]);
        }
        c[row * N + col] = sum;
    }
}

// Kernel 2: Tensor Core GEMM (using WMMA API)
__global__ void wmmaGemm(half* a, half* b, float* c, int M, int N, int K) {
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
        int aCol = (i + WMMA_K > K) ? K - i : WMMA_K;
        
        // Load fragments
        wmma::load_matrix_sync(fragA, a + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(fragB, b + i * N + warpN * WMMA_N, N);
        
        // Matrix multiply-accumulate
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // Store the result
    wmma::store_matrix_sync(c + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
}

// Kernel 3: Student Exercise - Implement custom tensor core operation
__global__ void studentTensorCoreOp(half* a, half* b, float* c, int M, int N, int K) {
    // TODO: Implement a custom tensor core operation using WMMA API
    // HINT: Use different tile sizes or implement a different operation
    
    // FIX: Define appropriate tensor core tile dimensions
    const int WMMA_M = /* YOUR CHOICE */;
    const int WMMA_N = /* YOUR CHOICE */;
    const int WMMA_K = /* YOUR CHOICE */;
    
    // FIX: Calculate warp and lane indices appropriately
    int warpM = /* YOUR CALCULATION */;
    int warpN = /* YOUR CALCULATION */;
    int lane = /* YOUR CALCULATION */;
    
    // Bounds checking
    if (warpM * WMMA_M >= M || warpN * WMMA_N >= N) {
        return;
    }
    
    // FIX: Allocate appropriate fragments for your operation
    // wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, half, wmma::row_major> fragA;
    // wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, half, wmma::col_major> fragB;
    // wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> fragC;
    
    // FIX: Initialize accumulator
    // wmma::fill_fragment(fragC, 0.0f);
    
    // FIX: Loop over K dimension and perform computation
    for (int i = 0; i < K; i += WMMA_K) {
        // FIX: Load matrices into fragments
        // wmma::load_matrix_sync(fragA, ...);
        // wmma::load_matrix_sync(fragB, ...);
        
        // FIX: Perform matrix multiply-accumulate
        // wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // FIX: Store the result
    // wmma::store_matrix_sync(...);
}

// Kernel 4: Student Exercise - Implement fused tensor core operation
__global__ void studentFusedTensorCore(half* a, half* b, float* c, float* bias, int M, int N, int K) {
    // TODO: Implement a fused operation that combines tensor core GEMM with bias addition
    // HINT: Perform GEMM and add bias in the same kernel
    
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
        wmma::load_matrix_sync(fragA, a + warpM * WMMA_M * K + i, K);
        wmma::load_matrix_sync(fragB, b + i * N + warpN * WMMA_N, N);
        
        // Matrix multiply-accumulate
        wmma::mma_sync(fragC, fragA, fragB, fragC);
    }
    
    // FIX: Add bias to the result before storing
    // HINT: You'll need to load bias values and add them to fragC
    // This requires element-wise operations on the fragment
    
    // Store the result
    wmma::store_matrix_sync(c + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
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
    printf("=== Tensor Cores Exercise ===\n");
    printf("Learn to use Tensor Cores for high-performance matrix operations.\n\n");

    // Setup parameters
    const int M = 256, N = 256, K = 256;  // Must be multiples of 8, 16, or 32 depending on tensor core size
    size_t bytes_a = M * K * sizeof(half);
    size_t bytes_b = K * N * sizeof(half);
    size_t bytes_c = M * N * sizeof(float);
    size_t bytes_bias = N * sizeof(float);
    
    // Allocate host memory
    half *h_a, *h_b;
    float *h_c_regular, *h_c_wmma, *h_c_student, *h_c_fused, *h_bias;
    
    h_a = (half*)malloc(bytes_a);
    h_b = (half*)malloc(bytes_b);
    h_c_regular = (float*)malloc(bytes_c);
    h_c_wmma = (float*)malloc(bytes_c);
    h_c_student = (float*)malloc(bytes_c);
    h_c_fused = (float*)malloc(bytes_c);
    h_bias = (float*)malloc(bytes_bias);
    
    // Initialize matrices
    initHalfMatrix(h_a, M, K, 0.1f);
    initHalfMatrix(h_b, K, N, 0.2f);
    initFloatMatrix(h_bias, 1, N, 0.1f);
    
    // Initialize output matrices to zero
    memset(h_c_regular, 0, bytes_c);
    memset(h_c_wmma, 0, bytes_c);
    memset(h_c_student, 0, bytes_c);
    memset(h_c_fused, 0, bytes_c);
    
    // Allocate device memory
    half *d_a, *d_b;
    float *d_c_regular, *d_c_wmma, *d_c_student, *d_c_fused, *d_bias;
    
    cudaMalloc(&d_a, bytes_a);
    cudaMalloc(&d_b, bytes_b);
    cudaMalloc(&d_c_regular, bytes_c);
    cudaMalloc(&d_c_wmma, bytes_c);
    cudaMalloc(&d_c_student, bytes_c);
    cudaMalloc(&d_c_fused, bytes_c);
    cudaMalloc(&d_bias, bytes_bias);
    
    // Copy matrices to device
    cudaMemcpy(d_a, h_a, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bytes_b, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bytes_bias, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    // Run regular GEMM kernel
    printf("Running regular GEMM kernel...\n");
    regularGemm<<<gridSize, blockSize>>>(d_a, d_b, d_c_regular, M, N, K);
    cudaDeviceSynchronize();
    
    // Run WMMA GEMM kernel
    printf("Running WMMA GEMM kernel...\n");
    dim3 wmmaBlockSize(32, 2);  // 32 threads per warp, 2 warps per block
    dim3 wmmaGridSize((N + 16 - 1) / 16, (M + 16 - 1) / 16);  // 16x16 tiles
    wmmaGemm<<<wmmaGridSize, wmmaBlockSize>>>(d_a, d_b, d_c_wmma, M, N, K);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student tensor core exercises (complete the code first!)...\n");
    
    // Custom tensor core operation exercise
    studentTensorCoreOp<<<wmmaGridSize, wmmaBlockSize>>>(d_a, d_b, d_c_student, M, N, K);
    cudaDeviceSynchronize();
    
    // Fused tensor core operation exercise
    studentFusedTensorCore<<<wmmaGridSize, wmmaBlockSize>>>(d_a, d_b, d_c_fused, d_bias, M, N, K);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the tensor core implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_c_regular, d_c_regular, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_wmma, d_c_wmma, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_student, d_c_student, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_c_fused, d_c_fused, bytes_c, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Regular GEMM: %.2f %.2f %.2f %.2f %.2f\n", 
           h_c_regular[0], h_c_regular[1], h_c_regular[2], h_c_regular[3], h_c_regular[4]);
    printf("WMMA GEMM:    %.2f %.2f %.2f %.2f %.2f\n", 
           h_c_wmma[0], h_c_wmma[1], h_c_wmma[2], h_c_wmma[3], h_c_wmma[4]);
    
    // Cleanup
    free(h_a); free(h_b); free(h_c_regular); free(h_c_wmma); 
    free(h_c_student); free(h_c_fused); free(h_bias);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c_regular); cudaFree(d_c_wmma);
    cudaFree(d_c_student); cudaFree(d_c_fused); cudaFree(d_bias);
    
    printf("\nExercise completed! Notice how Tensor Cores accelerate matrix operations.\n");
    
    return 0;
}