// ============================================================================
// Lesson 1.2: Vector Addition - Parallel Computing Fundamentals
// ============================================================================
// Concepts Covered:
//   - Memory allocation (cudaMalloc, cudaMemcpy)
//   - Thread indexing with offset
//   - Grid-stride loop pattern
//   - Error checking
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                   cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Vector addition kernel
// Each thread computes one element: C[i] = A[i] + B[i]
__global__ void vectorAdd(const float *A, const float *B, float *C, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Bounds check (handles cases where n is not multiple of block size)
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

// Alternative: Grid-stride loop (handles any array size efficiently)
__global__ void vectorAddGridStride(const float *A, const float *B, float *C, int n) {
    // Calculate global thread index
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid stride: total number of threads in grid
    int stride = blockDim.x * gridDim.x;
    
    // Process multiple elements per thread if needed
    for (int j = i; j < n; j += stride) {
        C[j] = A[j] + B[j];
    }
}

void initializeArray(float *arr, int n, float value) {
    for (int i = 0; i < n; i++) {
        arr[i] = value;
    }
}

int main() {
    int n = 10000;  // Vector size
    size_t size = n * sizeof(float);
    
    // Host (CPU) arrays
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    // Initialize input arrays
    initializeArray(h_A, n, 2.0f);  // A = [2, 2, 2, ...]
    initializeArray(h_B, n, 3.0f);  // B = [3, 3, 3, ...]
    
    // Device (GPU) arrays
    float *d_A, *d_B, *d_C;
    
    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Copy data from host to device
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));
    
    // Configure execution parameters
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Vector size: %d\n", n);
    printf("Blocks: %d, Threads per block: %d\n", blocksPerGrid, threadsPerBlock);
    printf("Launching kernel...\n\n");
    
    // Launch kernel
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);
    
    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());
    
    // Wait for GPU to complete
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    
    // Verify results (check first 10 elements)
    printf("Results (first 10 elements):\n");
    for (int i = 0; i < 10; i++) {
        printf("  C[%d] = %.1f (expected: 5.0)\n", i, h_C[i]);
    }
    
    // Verify all results
    bool correct = true;
    for (int i = 0; i < n; i++) {
        if (h_C[i] != 5.0f) {
            correct = false;
            break;
        }
    }
    printf("\nVerification: %s\n", correct ? "PASSED ✓" : "FAILED ✗");
    
    // Cleanup
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Memory Management:
//    - cudaMalloc(): Allocate device memory
//    - cudaMemcpy(): Transfer data host <-> device
//    - cudaFree(): Free device memory
//
// 2. Thread Indexing:
//    - threadIdx.x: Thread index within block (0 to blockDim.x-1)
//    - blockIdx.x:  Block index within grid
//    - blockDim.x:  Number of threads per block
//    - Global index = blockIdx.x * blockDim.x + threadIdx.x
//
// 3. Grid-stride loop:
//    - Allows handling arrays larger than total thread count
//    - Each thread processes multiple elements spaced by grid size
//
// EXERCISES:
// 1. Implement vector subtraction: C[i] = A[i] - B[i]
// 2. Implement element-wise multiplication: C[i] = A[i] * B[i]
// 3. Try different block sizes: 32, 64, 128, 256, 512, 1024
// 4. What happens if you remove the bounds check (if i < n)?
// 5. Implement the grid-stride version and compare results
// ============================================================================
