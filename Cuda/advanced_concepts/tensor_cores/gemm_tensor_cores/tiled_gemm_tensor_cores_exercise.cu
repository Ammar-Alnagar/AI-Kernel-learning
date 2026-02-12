/*
 * Tiled GEMM with Tensor Cores Exercise
 *
 * This exercise demonstrates how to implement optimized GEMM using both tiling
 * and Tensor Cores for maximum performance.
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <wmma.h>
#include <stdio.h>
#include <stdlib.h>

using namespace nvcuda;

// Kernel 1: Regular Tiled GEMM (without Tensor Cores)
__global__ void regularTiledGemm(half* A, half* B, float* C, int M, int N, int K) {
    #define TILE_SIZE 16
    __shared__ half As[TILE_SIZE][TILE_SIZE];
    __shared__ half Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    float Csub = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative loading of tiles into shared memory
        if (by * TILE_SIZE + ty < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[(by * TILE_SIZE + ty) * K + (t * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = __float2half(0.0f);
        }
        
        if (t * TILE_SIZE + ty < K && bx * TILE_SIZE + tx < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)];
        } else {
            Bs[ty][tx] = __float2half(0.0f);
        }
        
        __syncthreads();
        
        // Compute partial result for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            Csub += __half2float(As[ty][k]) * __half2float(Bs[k][tx]);
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    if (row < M && col < N) {
        C[row * N + col] = Csub;
    }
    #undef TILE_SIZE
}

// Kernel 2: Tensor Core GEMM (using WMMA API)
__global__ void tensorCoreGemm(half* A, half* B, float* C, int M, int N, int K) {
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
    
    // Store the result
    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
}

// Kernel 3: Student Exercise - Implement mixed-precision GEMM with Tensor Cores
__global__ void studentMixedPrecisionGemm(half* A, half* B, float* C, float* D, 
                                        int M, int N, int K, float alpha, float beta) {
    // TODO: Implement a mixed-precision GEMM that computes D = alpha * A * B + beta * C
    // using Tensor Cores for the A*B computation and then combining with C
    // HINT: Use WMMA API for A*B, then scale and add C in a single kernel
    
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
    
    // FIX: Apply alpha scaling to the result and add beta*C
    // HINT: You'll need to load C values and perform the computation: D = alpha*(A*B) + beta*C
    // This requires element-wise operations on the accumulator fragment
    
    // For now, just store the raw result
    wmma::store_matrix_sync(D + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
}

// Kernel 4: Student Exercise - Implement fused GEMM with activation using Tensor Cores
__global__ void studentFusedGemmActivation(half* A, half* B, float* C, 
                                         int M, int N, int K, float threshold) {
    // TODO: Implement fused GEMM + activation (e.g., ReLU) using Tensor Cores
    // HINT: Compute A*B using Tensor Cores, then apply activation function in the same kernel
    
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
    
    // FIX: Apply activation function (e.g., ReLU: max(0, x)) to the result
    // HINT: You'll need to iterate through the accumulator fragment and apply the activation
    // This requires accessing individual elements of the fragment
    
    // For now, just store the raw result
    wmma::store_matrix_sync(C + warpM * WMMA_M * N + warpN * WMMA_N, fragC, N, wmma::mem_row_major);
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
    printf("=== Tiled GEMM with Tensor Cores Exercise ===\n");
    printf("Learn to implement optimized GEMM using both tiling and Tensor Cores.\n\n");

    // Setup parameters
    const int M = 256, N = 256, K = 256;  // Must be multiples of 8, 16, or 32 depending on tensor core size
    size_t bytes_a = M * K * sizeof(half);
    size_t bytes_b = K * N * sizeof(half);
    size_t bytes_c = M * N * sizeof(float);
    
    // Allocate host memory
    half *h_A, *h_B;
    float *h_C_regular, *h_C_tensor, *h_C_mixed, *h_C_fused;
    
    h_A = (half*)malloc(bytes_a);
    h_B = (half*)malloc(bytes_b);
    h_C_regular = (float*)malloc(bytes_c);
    h_C_tensor = (float*)malloc(bytes_c);
    h_C_mixed = (float*)malloc(bytes_c);
    h_C_fused = (float*)malloc(bytes_c);
    
    // Initialize matrices
    initHalfMatrix(h_A, M, K, 0.1f);
    initHalfMatrix(h_B, K, N, 0.2f);
    
    // Initialize output matrices to zero
    memset(h_C_regular, 0, bytes_c);
    memset(h_C_tensor, 0, bytes_c);
    memset(h_C_mixed, 0, bytes_c);
    memset(h_C_fused, 0, bytes_c);
    
    // Allocate device memory
    half *d_A, *d_B;
    float *d_C_regular, *d_C_tensor, *d_C_mixed, *d_C_fused;
    
    cudaMalloc(&d_A, bytes_a);
    cudaMalloc(&d_B, bytes_b);
    cudaMalloc(&d_C_regular, bytes_c);
    cudaMalloc(&d_C_tensor, bytes_c);
    cudaMalloc(&d_C_mixed, bytes_c);
    cudaMalloc(&d_C_fused, bytes_c);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_b, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);  // 16x16 threads per block for regular tiled GEMM
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);  // 16x16 tiles
    
    // Run regular tiled GEMM kernel
    printf("Running regular tiled GEMM kernel...\n");
    regularTiledGemm<<<gridSize, blockSize>>>(d_A, d_B, d_C_regular, M, N, K);
    cudaDeviceSynchronize();
    
    // Run Tensor Core GEMM kernel
    printf("Running Tensor Core GEMM kernel...\n");
    dim3 tcBlockSize(32, 2);  // 32 threads per warp, 2 warps per block
    dim3 tcGridSize((N + 15) / 16, (M + 15) / 16);  // 16x16 tiles
    tensorCoreGemm<<<tcGridSize, tcBlockSize>>>(d_A, d_B, d_C_tensor, M, N, K);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student Tensor Core exercises (complete the code first!)...\n");
    
    // Mixed-precision GEMM exercise
    float alpha = 1.0f, beta = 1.0f;
    studentMixedPrecisionGemm<<<tcGridSize, tcBlockSize>>>(d_A, d_B, d_C_tensor, d_C_mixed, M, N, K, alpha, beta);
    cudaDeviceSynchronize();
    
    // Fused GEMM + activation exercise
    float threshold = 0.0f;
    studentFusedGemmActivation<<<tcGridSize, tcBlockSize>>>(d_A, d_B, d_C_fused, M, N, K, threshold);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the Tensor Core implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_C_regular, d_C_regular, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tensor, d_C_tensor, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_mixed, d_C_mixed, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_fused, d_C_fused, bytes_c, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Regular Tiled: %.2f %.2f %.2f %.2f %.2f\n", 
           h_C_regular[0], h_C_regular[1], h_C_regular[2], h_C_regular[3], h_C_regular[4]);
    printf("Tensor Core:   %.2f %.2f %.2f %.2f %.2f\n", 
           h_C_tensor[0], h_C_tensor[1], h_C_tensor[2], h_C_tensor[3], h_C_tensor[4]);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_regular); free(h_C_tensor); 
    free(h_C_mixed); free(h_C_fused);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_regular); cudaFree(d_C_tensor);
    cudaFree(d_C_mixed); cudaFree(d_C_fused);
    
    printf("\nExercise completed! Notice how Tensor Cores accelerate GEMM operations.\n");
    
    return 0;
}