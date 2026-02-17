// ============================================================================
// Lesson 3.1: Shared Memory Basics - On-Chip Communication
// ============================================================================
// Concepts Covered:
//   - Shared memory declaration and usage
//   - Thread block cooperation
//   - __syncthreads() barrier
//   - Shared memory benefits
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
// SHARED MEMORY DECLARATION
// __shared__ keyword declares shared memory
// Visible to all threads in the block
// Lifetime: kernel execution (block lifetime)
// ============================================================================

// Static shared memory (size known at compile time)
__global__ void sharedMemoryBasic(float *input, float *output, int n) {
    // Declare 256 floats of shared memory (one per thread in block)
    __shared__ float sharedData[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // Load from global to shared memory
        sharedData[tid] = input[idx];
        
        // SYNCHRONIZATION: Ensure all loads complete
        __syncthreads();
        
        // Now all threads can read any element in sharedData
        // Example: each thread computes sum with neighbor
        float neighborValue = (tid > 0) ? sharedData[tid - 1] : 0;
        output[idx] = sharedData[tid] + neighborValue;
    }
}

// ============================================================================
// SHARED MEMORY FOR REUSE
// Load data once, use multiple times
// Reduces global memory bandwidth
// ============================================================================
__global__ void sharedMemoryReuse(float *input, float *output, int n) {
    __shared__ float sharedData[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // Load once to shared memory
        sharedData[tid] = input[idx];
        __syncthreads();
        
        // Use the data multiple times (example: 3 operations)
        float val = sharedData[tid];
        float result = val * 2.0f;
        result = result + sharedData[(tid + 1) % blockDim.x];
        result = result * 0.5f;
        
        output[idx] = result;
    }
}

// ============================================================================
// SHARED MEMORY COMMUNICATION PATTERN
// Threads exchange data through shared memory
// ============================================================================
__global__ void sharedMemoryCommunication(float *input, float *output, int n) {
    __shared__ float sharedData[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        // Each thread loads its element
        sharedData[tid] = input[idx];
        __syncthreads();
        
        // Now threads can read each other's data
        // Example: reverse the data within block
        int reverseIdx = blockDim.x - 1 - tid;
        output[idx] = sharedData[reverseIdx];
    }
}

// ============================================================================
// DYNAMIC SHARED MEMORY
// Size determined at kernel launch time
// Useful for flexible block sizes
// ============================================================================
__global__ void sharedMemoryDynamic(float *input, float *output, int n) {
    // Dynamic shared memory declaration (size specified at launch)
    extern __shared__ float sharedData[];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    if (idx < n) {
        sharedData[tid] = input[idx];
        __syncthreads();
        
        output[idx] = sharedData[tid] * 2.0f;
    }
}

// ============================================================================
// BLOCK REDUCTION USING SHARED MEMORY
// Classic parallel reduction pattern
// ============================================================================
__global__ void blockReduction(float *input, float *output, int n) {
    __shared__ float sharedData[256];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data
    sharedData[tid] = (idx < n) ? input[idx] : 0;
    __syncthreads();
    
    // Parallel reduction in shared memory
    // Each step halves the number of active threads
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();  // Critical! Ensure all adds complete
    }
    
    // Thread 0 writes block's result
    if (tid == 0) {
        output[blockIdx.x] = sharedData[0];
    }
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = 1.0f;  // All ones for easy verification
    }
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("=== Shared Memory Demonstrations ===\n\n");
    
    // Basic shared memory
    printf("1. Basic Shared Memory:\n");
    sharedMemoryBasic<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("   First 5 elements: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", h_output[i]);
    printf("\n   (Each element = itself + left neighbor)\n\n");
    
    // Communication pattern
    printf("2. Communication Pattern (reverse within block):\n");
    sharedMemoryCommunication<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("   Kernel executed successfully\n\n");
    
    // Dynamic shared memory
    printf("3. Dynamic Shared Memory:\n");
    size_t sharedMemSize = threadsPerBlock * sizeof(float);
    sharedMemoryDynamic<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("   First 5 elements: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", h_output[i]);
    printf("\n   (Each element * 2)\n\n");
    
    // Block reduction
    printf("4. Block Reduction (sum within each block):\n");
    float *d_blockSums;
    cudaMalloc(&d_blockSums, blocksPerGrid * sizeof(float));
    blockReduction<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_blockSums, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float *h_blockSums = (float *)malloc(blocksPerGrid * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));
    
    float totalSum = 0;
    for (int i = 0; i < blocksPerGrid; i++) {
        totalSum += h_blockSums[i];
    }
    printf("   Number of blocks: %d\n", blocksPerGrid);
    printf("   Sum per block (first 5): ");
    for (int i = 0; i < 5; i++) printf("%.0f ", h_blockSums[i]);
    printf("\n   Total sum: %.0f (expected: %d)\n", totalSum, n);
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_blockSums);
    free(h_input);
    free(h_output);
    free(h_blockSums);
    
    printf("\n=== Shared Memory Summary ===\n");
    printf("| Feature        | Details                                    |\n");
    printf("|----------------|--------------------------------------------|\n");
    printf("| Location       | On-chip (much faster than global)          |\n");
    printf("| Scope          | Thread block                               |\n");
    printf("| Lifetime       | Kernel execution                           |\n");
    printf("| Size Limit     | ~48KB per block (configurable)             |\n");
    printf("| Latency        | ~100x faster than global memory            |\n");
    printf("| Synchronization| __syncthreads() required                   |\n");
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Shared Memory Characteristics:
//    - On-chip memory (very fast, low latency)
//    - Shared by all threads in a block
//    - Must synchronize with __syncthreads()
//    - Limited size (~48KB per SM)
//
// 2. __syncthreads():
//    - Block-wide barrier
//    - All threads must reach it
//    - Ensures memory operations complete
//    - Cannot be in conditional code!
//
// 3. Use Cases:
//    - Data reuse (load once, use many times)
//    - Thread communication
//    - Reduction operations
//    - Tiled algorithms (matrix multiply)
//
// EXERCISES:
// 1. Implement a prefix sum (scan) using shared memory
// 2. Create a block-level histogram using shared memory
// 3. What happens if you remove __syncthreads()?
// 4. Research: What are bank conflicts in shared memory?
// 5. Try different block sizes: 32, 64, 128, 256, 512
// ============================================================================
