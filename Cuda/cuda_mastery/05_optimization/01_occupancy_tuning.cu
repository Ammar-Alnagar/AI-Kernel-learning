// ============================================================================
// Lesson 5.1: Occupancy Tuning - Maximizing GPU Utilization
// ============================================================================
// Concepts Covered:
//   - What is occupancy?
//   - Resource limits (registers, shared memory)
//   - Launch configuration optimization
//   - CUDA Occupancy API
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
// LOW OCCUPANCY KERNEL (uses many registers)
// Heavy register usage limits concurrent blocks
// ============================================================================
__global__ void lowOccupancyKernel(float *data, int n) {
    // Force high register usage with many local variables
    register float r0, r1, r2, r3, r4, r5, r6, r7;
    register float r8, r9, r10, r11, r12, r13, r14, r15;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Use all registers to prevent optimization
        r0 = data[idx];
        r1 = r0 * 2.0f;
        r2 = r1 + 1.0f;
        r3 = r2 * 3.0f;
        r4 = r3 - 2.0f;
        r5 = r4 * 4.0f;
        r6 = r5 + 3.0f;
        r7 = r6 * 5.0f;
        r8 = r7 - 4.0f;
        r9 = r8 * 6.0f;
        r10 = r9 + 5.0f;
        r11 = r10 * 7.0f;
        r12 = r11 - 6.0f;
        r13 = r12 * 8.0f;
        r14 = r13 + 7.0f;
        r15 = r14 * 9.0f;
        
        data[idx] = r15;
    }
}

// ============================================================================
// HIGH OCCUPANCY KERNEL (minimal register usage)
// Light register usage allows more concurrent blocks
// ============================================================================
__global__ void highOccupancyKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Minimal register usage
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// ============================================================================
// SHARED MEMORY LIMITED KERNEL
// High shared memory usage limits occupancy
// ============================================================================
__global__ void sharedMemLimitedKernel(float *data, int n) {
    // Large shared memory allocation
    __shared__ float sharedData[4096];  // 16KB per block
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        sharedData[threadIdx.x] = data[idx];
        __syncthreads();
        
        data[idx] = sharedData[threadIdx.x] * 2.0f;
    }
}

void printDeviceProperties() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    printf("=== Device Properties ===\n");
    printf("Device: %s\n", prop.name);
    printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
    printf("Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("Max registers per SM: %d\n", prop.regsPerMultiprocessor);
    printf("Max shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("Warp size: %d\n\n", prop.warpSize);
}

void analyzeOccupancy(void (*kernel)(float*, int), const char *name, 
                      int threadsPerBlock, int n) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    
    int minGridSize, blockSize;
    
    // Get occupancy info
    int minBlocksPerSM;
    cudaOccupancyMaxPotentialBlockSize(&minBlocksPerSM, &blockSize, kernel, 0, 0);
    
    // Calculate theoretical occupancy
    int maxActiveBlocksPerSM = prop.maxThreadsPerMultiProcessor / blockSize;
    float occupancy = (float)maxActiveBlocksPerSM / 
                      (prop.maxThreadsPerMultiProcessor / prop.maxThreadsPerBlock);
    
    printf("=== %s ===\n", name);
    printf("Requested threads/block: %d\n", threadsPerBlock);
    printf("Optimal block size (occupancy API): %d\n", blockSize);
    printf("Min blocks per SM for occupancy: %d\n", minBlocksPerSM);
    printf("Max active blocks per SM: %d\n", maxActiveBlocksPerSM);
    printf("Theoretical occupancy: %.1f%%\n\n", occupancy * 100);
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t size = n * sizeof(float);
    
    float *h_data = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = 1.0f;
    }
    
    float *d_data;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    printDeviceProperties();
    
    // Analyze different kernels
    analyzeOccupancy(lowOccupancyKernel, "Low Occupancy Kernel (High Regs)", 256, n);
    analyzeOccupancy(highOccupancyKernel, "High Occupancy Kernel (Low Regs)", 256, n);
    analyzeOccupancy(sharedMemLimitedKernel, "Shared Memory Limited Kernel", 256, n);
    
    // Performance comparison
    printf("=== Performance Comparison ===\n\n");
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    // Low occupancy kernel
    lowOccupancyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        lowOccupancyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float lowTime;
    cudaEventElapsedTime(&lowTime, start, stop);
    printf("Low Occupancy Kernel:  %.3f ms (100 iterations)\n", lowTime);
    
    // High occupancy kernel
    highOccupancyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        highOccupancyKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float highTime;
    cudaEventElapsedTime(&highTime, start, stop);
    printf("High Occupancy Kernel: %.3f ms (100 iterations)\n\n", highTime);
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    free(h_data);
    
    printf("=== Occupancy Optimization Tips ===\n");
    printf("1. Use cudaOccupancyMaxPotentialBlockSize() to find optimal config\n");
    printf("2. Reduce register usage with -maxrregcount compiler flag\n");
    printf("3. Minimize shared memory per block\n");
    printf("4. Increase block size (up to 512-1024) for better occupancy\n");
    printf("5. Higher occupancy != always faster (depends on workload)\n");
    printf("6. Memory-bound kernels benefit less from high occupancy\n");
    printf("7. Compute-bound kernels benefit more from high occupancy\n");
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Occupancy Definition:
//    - Ratio of active warps to maximum possible warps
//    - Higher occupancy = more latency hiding
//    - Limited by registers, shared memory, threads
//
// 2. Resource Limits:
//    - Registers per SM: 65536 (typical)
//    - Shared memory per SM: 48-96 KB
//    - Max threads per SM: 2048
//    - Max threads per block: 1024
//
// 3. Occupancy Calculation:
//    - Active blocks limited by most constrained resource
//    - Occupancy = active_warps / max_warps
//
// 4. When High Occupancy Matters:
//    - Compute-bound kernels
//    - Kernels with high latency (memory access)
//    - Less important for already-fast kernels
//
// EXERCISES:
// 1. Use nvprof to measure achieved occupancy
// 2. Try -maxrregcount=32 and compare performance
// 3. Find the optimal block size for your kernel
// 4. Research: What is "latency hiding"?
// 5. Profile with Nsight Compute to see bottlenecks
// ============================================================================
