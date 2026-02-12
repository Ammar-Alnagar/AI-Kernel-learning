/*
 * GEMM (General Matrix Multiplication) Exercise
 *
 * This exercise demonstrates how to implement optimized General Matrix Multiplication
 * using shared memory tiling and other optimization techniques.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

// Kernel 1: Naive GEMM (INEFFICIENT)
__global__ void naiveGemm(float alpha, float* A, float* B, float beta, float* C, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Kernel 2: Tiled GEMM (EFFICIENT)
__global__ void tiledGemm(float alpha, float* A, float* B, float beta, float* C, int M, int N, int K) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    float Csub = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Collaborative loading of tiles into shared memory
        if (by * TILE_SIZE + ty < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[(by * TILE_SIZE + ty) * K + (t * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (t * TILE_SIZE + ty < K && bx * TILE_SIZE + tx < N) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + (bx * TILE_SIZE + tx)];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial result for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    if (row < M && col < N) {
        C[row * N + col] = alpha * Csub + beta * C[row * N + col];
    }
}

// Kernel 3: Student Exercise - Implement coarsened GEMM
__global__ void studentCoarsenedGemm(float alpha, float* A, float* B, float beta, float* C, int M, int N, int K) {
    // TODO: Implement a coarsened version where each thread computes multiple output elements
    // HINT: Have each thread compute multiple C elements by using a stride pattern
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = M * N;
    
    // FIX: Implement coarsened computation where each thread processes multiple elements
    // Use a stride pattern similar to the coarsened kernels from previous exercises
    for (int idx = /* YOUR START INDEX */; idx < total_elements; idx += /* YOUR STRIDE */) {
        int row = /* CONVERT IDX TO ROW */;
        int col = /* CONVERT IDX TO COL */;
        
        if (row < M && col < N) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            C[row * N + col] = alpha * sum + beta * C[row * N + col];
        }
    }
}

// Kernel 4: Student Exercise - Implement optimized GEMM with register blocking
__global__ void studentOptimizedGemm(float alpha, float* A, float* B, float beta, float* C, int M, int N, int K) {
    // TODO: Implement GEMM with register-level blocking for even better performance
    // HINT: Load multiple values from shared memory into registers and compute multiple outputs
    
    // Define register blocking factors (small constants)
    const int REG_M = 2;  // Process 2 rows per thread
    const int REG_N = 2;  // Process 2 cols per thread
    
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // FIX: Implement register blocking
    // Pre-load values into registers and compute multiple output elements
    // This reduces shared memory accesses and increases arithmetic intensity
    
    // Example structure (you'll need to implement the full logic):
    /*
    float C_reg[REG_M][REG_N] = {0};  // Register array for partial results
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory (same as before)
        // ...
        
        __syncthreads();
        
        // Compute using register blocking
        for (int k = 0; k < TILE_SIZE; k++) {
            // Load values from shared memory to registers
            float a_vals[REG_M];
            float b_vals[REG_N];
            
            // Load A values
            #pragma unroll
            for (int rm = 0; rm < REG_M; rm++) {
                if (by * TILE_SIZE + ty + rm * blockDim.y < M) {
                    a_vals[rm] = As[ty + rm * blockDim.y][k];  // This indexing needs adjustment
                }
            }
            
            // Load B values
            #pragma unroll
            for (int rn = 0; rn < REG_N; rn++) {
                if (bx * TILE_SIZE + tx + rn * blockDim.x < N) {
                    b_vals[rn] = Bs[k][tx + rn * blockDim.x];  // This indexing needs adjustment
                }
            }
            
            // Accumulate results in registers
            #pragma unroll
            for (int rm = 0; rm < REG_M; rm++) {
                #pragma unroll
                for (int rn = 0; rn < REG_N; rn++) {
                    C_reg[rm][rn] += a_vals[rm] * b_vals[rn];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results back to global memory
    #pragma unroll
    for (int rm = 0; rm < REG_M; rm++) {
        #pragma unroll
        for (int rn = 0; rn < REG_N; rn++) {
            int row = by * TILE_SIZE + ty + rm * /* appropriate offset */;
            int col = bx * TILE_SIZE + tx + rn * /* appropriate offset */;
            if (row < M && col < N) {
                C[row * N + col] = alpha * C_reg[rm][rn] + beta * C[row * N + col];
            }
        }
    }
    */
    
    // PLACEHOLDER - REPLACE WITH YOUR IMPLEMENTATION
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int rows, int cols, float start_val = 1.0f) {
    for (int i = 0; i < rows * cols; i++) {
        mat[i] = start_val + (i % 100) * 0.01f;
    }
}

// Utility function to print matrix
void printMatrix(float* mat, int rows, int cols, int print_rows = 3, int print_cols = 3) {
    printf("First %d×%d elements:\n", print_rows, print_cols);
    for (int i = 0; i < print_rows && i < rows; i++) {
        for (int j = 0; j < print_cols && j < cols; j++) {
            printf("%6.2f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== GEMM (General Matrix Multiplication) Exercise ===\n");
    printf("Learn to implement optimized matrix multiplication kernels.\n\n");

    // Setup parameters
    const int M = 512, N = 512, K = 512;  // Matrix dimensions
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A, *h_B, *h_C_naive, *h_C_tiled, *h_C_coarsened, *h_C_optimized;
    h_A = (float*)malloc(bytes_A);
    h_B = (float*)malloc(bytes_B);
    h_C_naive = (float*)malloc(bytes_C);
    h_C_tiled = (float*)malloc(bytes_C);
    h_C_coarsened = (float*)malloc(bytes_C);
    h_C_optimized = (float*)malloc(bytes_C);
    
    // Initialize matrices
    initMatrix(h_A, M, K, 1.0f);
    initMatrix(h_B, K, N, 2.0f);
    initMatrix(h_C_naive, M, N, 0.0f);  // Initialize C to zero for both kernels
    
    // Copy initial C to other matrices
    memcpy(h_C_tiled, h_C_naive, bytes_C);
    memcpy(h_C_coarsened, h_C_naive, bytes_C);
    memcpy(h_C_optimized, h_C_naive, bytes_C);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled, *d_C_coarsened, *d_C_optimized;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_C_naive, bytes_C);
    cudaMalloc(&d_C_tiled, bytes_C);
    cudaMalloc(&d_C_coarsened, bytes_C);
    cudaMalloc(&d_C_optimized, bytes_C);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_naive, h_C_naive, bytes_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_tiled, h_C_tiled, bytes_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_coarsened, h_C_coarsened, bytes_C, cudaMemcpyHostToDevice);
    cudaMemcpy(d_C_optimized, h_C_optimized, bytes_C, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((N + TILE_SIZE - 1) / TILE_SIZE, 
                  (M + TILE_SIZE - 1) / TILE_SIZE);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Run naive GEMM kernel
    printf("Running naive GEMM kernel...\n");
    naiveGemm<<<gridSize, blockSize>>>(alpha, d_A, d_B, beta, d_C_naive, M, N, K);
    cudaDeviceSynchronize();
    
    // Run tiled GEMM kernel
    printf("Running tiled GEMM kernel...\n");
    tiledGemm<<<gridSize, blockSize>>>(alpha, d_A, d_B, beta, d_C_tiled, M, N, K);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student GEMM exercises (complete the code first!)...\n");
    
    // Coarsened GEMM exercise
    int linearSize = M * N;
    int linearBlockSize = 256;
    int linearGridSize = (linearSize + linearBlockSize - 1) / linearBlockSize;
    studentCoarsenedGemm<<<linearGridSize, linearBlockSize>>>(alpha, d_A, d_B, beta, d_C_coarsened, M, N, K);
    cudaDeviceSynchronize();
    
    // Optimized GEMM exercise
    studentOptimizedGemm<<<gridSize, blockSize>>>(alpha, d_A, d_B, beta, d_C_optimized, M, N, K);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the GEMM implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_C_naive, d_C_naive, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tiled, d_C_tiled, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_coarsened, d_C_coarsened, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_optimized, d_C_optimized, bytes_C, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results from GEMM (first 3×3 elements):\n");
    printf("Naive GEMM result:\n");
    printMatrix(h_C_naive, M, N, 3, 3);
    printf("\nTiled GEMM result:\n");
    printMatrix(h_C_tiled, M, N, 3, 3);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); 
    free(h_C_coarsened); free(h_C_optimized);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_naive); cudaFree(d_C_tiled);
    cudaFree(d_C_coarsened); cudaFree(d_C_optimized);
    
    printf("\nExercise completed! Notice how tiling improves GEMM performance.\n");
    
    return 0;
}