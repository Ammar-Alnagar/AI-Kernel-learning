// ============================================================================
// Lesson 1.3: Thread Indexing - Mastering CUDA's Coordinate System
// ============================================================================
// Concepts Covered:
//   - 1D, 2D, 3D thread indexing
//   - blockIdx, threadIdx, blockDim, gridDim
//   - Multi-dimensional grid and block configurations
//   - Flattening multi-dimensional indices
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// 1D Indexing
// ============================================================================
__global__ void print1DIndex() {
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int globalId = bid * blockDim.x + tid;
    
    printf("Block %d, Thread %d, Global ID: %d\n", bid, tid, globalId);
}

// ============================================================================
// 2D Indexing - 2D Block within 1D Grid
// ============================================================================
__global__ void print2DBlockIndex() {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bid = blockIdx.x;
    
    // 2D thread index within block
    int tid2D = ty * blockDim.x + tx;
    
    printf("Block %d, Thread (%d, %d), 1D ID: %d\n", bid, tx, ty, tid2D);
}

// ============================================================================
// 2D Indexing - 2D Grid and 2D Block
// ============================================================================
__global__ void print2DGridIndex() {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    
    // Calculate global 2D position
    int globalX = bx * blockDim.x + tx;
    int globalY = by * blockDim.y + ty;
    
    // Flatten to 1D (row-major order)
    int width = blockDim.x * gridDim.x;
    int globalId = globalY * width + globalX;
    
    printf("Block (%d, %d), Thread (%d, %d), Global (%d, %d), ID: %d\n",
           bx, by, tx, ty, globalX, globalY, globalId);
}

// ============================================================================
// 3D Indexing - For 3D data (volumes, images with batches)
// ============================================================================
__global__ void print3DIndex(int volumeWidth, int volumeHeight, int volumeDepth) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int tz = threadIdx.z;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int bz = blockIdx.z;
    
    // Global 3D coordinates
    int x = bx * blockDim.x + tx;
    int y = by * blockDim.y + ty;
    int z = bz * blockDim.z + tz;
    
    // Flatten to 1D index (row-major, z is slowest)
    int globalId = z * (volumeHeight * volumeWidth) + 
                   y * volumeWidth + 
                   x;
    
    printf("Thread (%d,%d,%d) -> Global (%d,%d,%d) -> ID: %d\n",
           tx, ty, tz, x, y, z, globalId);
}

// ============================================================================
// Matrix indexing example - Common pattern for 2D data
// ============================================================================
__global__ void matrixIndexDemo(float *matrix, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // x = column
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // y = row
    
    if (col < width && row < height) {
        // Row-major order: row * width + col
        int idx = row * width + col;
        matrix[idx] = row * 100 + col;  // Store position info
    }
}

int main() {
    printf("=== 1D Indexing Demo ===\n");
    printf("Configuration: 2 blocks, 4 threads per block\n\n");
    print1DIndex<<<2, 4>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n=== 2D Block Indexing Demo ===\n");
    printf("Configuration: 2 blocks, (2x2) threads per block\n\n");
    dim3 blockSize2D(2, 2);  // 2D block: 2 threads in x, 2 in y
    print2DBlockIndex<<<2, blockSize2D>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n=== 2D Grid Indexing Demo ===\n");
    printf("Configuration: (2x2) blocks, (2x2) threads per block\n\n");
    dim3 gridSize2D(2, 2);   // 2D grid
    dim3 blockSize2DGrid(2, 2);  // 2D block
    print2DGridIndex<<<gridSize2D, blockSize2DGrid>>>();
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n=== 3D Indexing Demo ===\n");
    printf("Configuration: (2x2x1) blocks, (2x2x2) threads per block\n\n");
    dim3 gridSize3D(2, 2, 1);
    dim3 blockSize3D(2, 2, 2);
    print3DIndex<<<gridSize3D, blockSize3D>>>(4, 4, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n=== Matrix Indexing Demo ===\n");
    int width = 8, height = 8;
    float *d_matrix;
    cudaMalloc(&d_matrix, width * height * sizeof(float));
    
    dim3 blockSize(4, 4);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
                  (height + blockSize.y - 1) / blockSize.y);
    
    matrixIndexDemo<<<gridSize, blockSize>>>(d_matrix, width, height);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("Matrix kernel launched successfully (%dx%d matrix)\n", width, height);
    
    cudaFree(d_matrix);
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Built-in Variables:
//    - threadIdx.{x,y,z}: Thread coordinates within block
//    - blockIdx.{x,y,z}:  Block coordinates within grid
//    - blockDim.{x,y,z}:  Block dimensions (threads per block)
//    - gridDim.{x,y,z}:   Grid dimensions (blocks per grid)
//
// 2. dim3 Type:
//    - Used for 2D/3D configurations
//    - dim3 var(x, y, z) - unspecified components default to 1
//
// 3. Index Flattening:
//    - 2D: idx = y * width + x  (row-major)
//    - 3D: idx = z * (height*width) + y * width + x
//
// 4. Common Patterns:
//    - 1D data: <<<blocks, threads>>>
//    - 2D data: <<<dim3(bx, by), dim3(tx, ty)>>>
//    - 3D data: <<<dim3(bx, by, bz), dim3(tx, ty, tz)>>>
//
// EXERCISES:
// 1. Create a kernel that processes a 16x16 image using 2D indexing
// 2. Implement a 3D volume initialization (e.g., set each voxel to x+y+z)
// 3. What's the maximum threads per block in 2D? (tx * ty * tz <= 1024)
// 4. Write a kernel that prints the total number of threads in the grid
// 5. Implement column-major ordering and compare with row-major
// ============================================================================
