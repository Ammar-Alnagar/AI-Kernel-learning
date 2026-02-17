// ============================================================================
// Lesson 3.2: Tiled Matrix Multiplication - Shared Memory Optimization
// ============================================================================
// Concepts Covered:
//   - Matrix multiplication basics
//   - Tiling strategy for data reuse
//   - Shared memory optimization
//   - Performance comparison: naive vs tiled
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <time.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

#define TILE_SIZE 16  // Must match block dimension

// ============================================================================
// NAIVE MATRIX MULTIPLICATION
// No shared memory - each element loaded multiple times from global memory
// C[i,j] = sum_k(A[i,k] * B[k,j])
// ============================================================================
__global__ void matrixMulNaive(float *A, float *B, float *C, int width) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (col < width && row < width) {
        float sum = 0.0f;
        
        // Dot product of row from A and column from B
        for (int k = 0; k < width; k++) {
            // Each thread loads A[row,k] and B[k,col] from global memory
            // These values are NOT reused - terrible bandwidth usage!
            sum += A[row * width + k] * B[k * width + col];
        }
        
        C[row * width + col] = sum;
    }
}

// ============================================================================
// TILED MATRIX MULTIPLICATION
// Uses shared memory to reuse loaded elements
// Each thread block computes one tile of the output matrix
// ============================================================================
__global__ void matrixMulTiled(float *A, float *B, float *C, int width) {
    // Shared memory tiles - each block has its own copy
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0.0f;
    
    // Loop over tiles
    // Number of tiles needed to cover the matrix
    int numTiles = (width + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        // Load one tile of A and B into shared memory
        // Each thread loads one element
        
        // Column index for A tile
        int tileCol = t * TILE_SIZE + threadIdx.x;
        
        // Row index for B tile
        int tileRow = t * TILE_SIZE + threadIdx.y;
        
        // Load with bounds checking
        if (col < width && tileRow < width) {
            As[threadIdx.y][threadIdx.x] = A[row * width + tileCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tileRow < width && col < width) {
            Bs[threadIdx.y][threadIdx.x] = B[tileRow * width + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        // Wait for all threads to load their tiles
        __syncthreads();
        
        // Compute partial dot product for this tile
        // Each thread has access to the full tile via shared memory
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        // Wait for all threads to finish using this tile
        __syncthreads();
    }
    
    // Write result
    if (col < width && row < width) {
        C[row * width + col] = sum;
    }
}

// ============================================================================
// OPTIMIZED TILED MATRIX MULTIPLICATION
// Reduces shared memory bank conflicts with padding
// ============================================================================
__global__ void matrixMulTiledOptimized(float *A, float *B, float *C, int width) {
    // Add padding to avoid bank conflicts
    // TILE_SIZE + 1 prevents conflicts in column access
    __shared__ float As[TILE_SIZE][TILE_SIZE + 1];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE + 1];
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    float sum = 0.0f;
    int numTiles = (width + TILE_SIZE - 1) / TILE_SIZE;
    
    for (int t = 0; t < numTiles; t++) {
        int tileCol = t * TILE_SIZE + threadIdx.x;
        int tileRow = t * TILE_SIZE + threadIdx.y;
        
        if (col < width && tileRow < width) {
            As[threadIdx.y][threadIdx.x] = A[row * width + tileCol];
        } else {
            As[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        if (tileRow < width && col < width) {
            Bs[threadIdx.y][threadIdx.x] = B[tileRow * width + col];
        } else {
            Bs[threadIdx.y][threadIdx.x] = 0.0f;
        }
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        
        __syncthreads();
    }
    
    if (col < width && row < width) {
        C[row * width + col] = sum;
    }
}

// Initialize matrix with simple pattern
void initMatrix(float *M, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            M[i * width + j] = (i + j) % 10 * 0.1f;  // Simple pattern
        }
    }
}

// Verify result
bool verifyResult(float *C, float *expected, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float diff = C[i * width + j] - expected[i * width + j];
            if (diff < 0) diff = -diff;
            if (diff > 0.01f) return false;
        }
    }
    return true;
}

// CPU reference
void matrixMulCPU(float *A, float *B, float *C, int width) {
    for (int i = 0; i < width; i++) {
        for (int j = 0; j < width; j++) {
            float sum = 0.0f;
            for (int k = 0; k < width; k++) {
                sum += A[i * width + k] * B[k * width + j];
            }
            C[i * width + j] = sum;
        }
    }
}

int main() {
    int width = 512;  // Matrix dimension (width x width)
    size_t size = width * width * sizeof(float);
    
    printf("Matrix Multiplication: %d x %d\n", width, width);
    printf("Tile size: %d x %d\n\n", TILE_SIZE, TILE_SIZE);
    
    // Host matrices
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C_naive = (float *)malloc(size);
    float *h_C_tiled = (float *)malloc(size);
    float *h_C_tiled_opt = (float *)malloc(size);
    float *h_C_ref = (float *)malloc(size);
    
    initMatrix(h_A, width);
    initMatrix(h_B, width);
    
    // Device matrices
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Execution config
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((width + TILE_SIZE - 1) / TILE_SIZE,
                  (width + TILE_SIZE - 1) / TILE_SIZE);
    
    printf("Grid: (%d, %d) blocks\n", gridSize.x, gridSize.y);
    printf("Block: (%d, %d) threads\n\n", blockSize.x, blockSize.y);
    
    // Create events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Naive version
    printf("Running Naive Matrix Multiply...\n");
    cudaEventRecord(start);
    matrixMulNaive<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float naiveTime;
    cudaEventElapsedTime(&naiveTime, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C_naive, d_C, size, cudaMemcpyDeviceToHost));
    printf("Naive time: %.3f ms\n\n", naiveTime);
    
    // Tiled version
    printf("Running Tiled Matrix Multiply...\n");
    cudaEventRecord(start);
    matrixMulTiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiledTime;
    cudaEventElapsedTime(&tiledTime, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C_tiled, d_C, size, cudaMemcpyDeviceToHost));
    printf("Tiled time: %.3f ms\n", tiledTime);
    printf("Speedup: %.2fx\n\n", naiveTime / tiledTime);
    
    // Optimized tiled version
    printf("Running Optimized Tiled Matrix Multiply...\n");
    cudaEventRecord(start);
    matrixMulTiledOptimized<<<gridSize, blockSize>>>(d_A, d_B, d_C, width);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float tiledOptTime;
    cudaEventElapsedTime(&tiledOptTime, start, stop);
    CUDA_CHECK(cudaMemcpy(h_C_tiled_opt, d_C, size, cudaMemcpyDeviceToHost));
    printf("Optimized Tiled time: %.3f ms\n", tiledOptTime);
    
    // Verify correctness
    printf("\n=== Verification ===\n");
    matrixMulCPU(h_A, h_B, h_C_ref, width);
    printf("Naive vs CPU: %s\n", verifyResult(h_C_naive, h_C_ref, width) ? "PASS" : "FAIL");
    printf("Tiled vs CPU: %s\n", verifyResult(h_C_tiled, h_C_ref, width) ? "PASS" : "FAIL");
    printf("Optimized vs CPU: %s\n", verifyResult(h_C_tiled_opt, h_C_ref, width) ? "PASS" : "FAIL");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C_naive);
    free(h_C_tiled);
    free(h_C_tiled_opt);
    free(h_C_ref);
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Tiling Strategy:
//    - Divide matrix into tiles that fit in shared memory
//    - Each block processes one output tile
//    - Load tiles cooperatively, compute, repeat
//
// 2. Data Reuse:
//    - Naive: Each element loaded N times from global memory
//    - Tiled: Each element loaded N/TILE_SIZE times
//    - Reduces global memory bandwidth by TILE_SIZE
//
// 3. Bank Conflicts:
//    - Shared memory divided into banks
//    - Simultaneous access to same bank = serialization
//    - Padding (TILE_SIZE+1) avoids conflicts
//
// EXERCISES:
// 1. Try different tile sizes: 8, 16, 32
// 2. What's the optimal tile size for your GPU?
// 3. Implement 1D tiling for vector operations
// 4. Add double buffering to hide memory latency
// 5. Research: How does tensor core acceleration work?
// ============================================================================
