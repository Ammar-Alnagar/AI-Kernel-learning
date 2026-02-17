// ============================================================================
// Lesson 6.2: Unified Memory - Simplified Memory Management
// ============================================================================
// Concepts Covered:
//   - cudaMallocManaged
//   - Automatic data migration
//   - Memory prefetching
//   - Unified memory best practices
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
// BASIC UNIFIED MEMORY KERNEL
// Same pointer accessible from CPU and GPU
// ============================================================================
__global__ void unifiedMemoryKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// CPU function to process same data
void cpuProcess(float *data, int n) {
    for (int i = 0; i < n; i++) {
        data[i] = data[i] + 10.0f;
    }
}

// ============================================================================
// IRREGULAR ACCESS PATTERN
// Unified memory excels with unpredictable access patterns
// ============================================================================
__global__ void irregularAccessKernel(float *data, int *indices, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Indirect access - hard to prefetch
        int accessIdx = indices[idx] % n;
        output[idx] = data[accessIdx] * 2.0f;
    }
}

// ============================================================================
// MEMORY ADVICE FOR PREFETCHING
// Hint to CUDA where memory will be accessed
// ============================================================================
__global__ void prefetchedKernel(float *data, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        output[idx] = data[idx] * data[idx];
    }
}

int main() {
    int n = 1000000;  // 1M elements
    size_t size = n * sizeof(float);
    
    printf("=== Unified Memory Demo ===\n\n");
    
    // Device info
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Managed Memory Support: %s\n\n", 
           prop.managedMemory ? "Yes" : "No");
    
    // ========================================================================
    // BASIC UNIFIED MEMORY USAGE
    // ========================================================================
    printf("1. Basic Unified Memory:\n");
    printf("   Allocating %zu MB with cudaMallocManaged...\n\n", size / (1024 * 1024));
    
    float *um_data;
    CUDA_CHECK(cudaMallocManaged(&um_data, size));
    
    // Initialize on CPU
    printf("   Initializing on CPU...\n");
    for (int i = 0; i < n; i++) {
        um_data[i] = 1.0f;
    }
    
    // Process on GPU
    printf("   Processing on GPU...\n");
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    unifiedMemoryKernel<<<blocksPerGrid, threadsPerBlock>>>(um_data, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Access again on CPU
    printf("   Verifying on CPU...\n");
    float sum = 0;
    for (int i = 0; i < 10; i++) {
        sum += um_data[i];
    }
    printf("   First 10 elements sum: %.1f (expected: 30.0)\n\n", sum);
    
    // ========================================================================
    // CPU-GPU SHARING
    // Same data accessed by both processors
    // ========================================================================
    printf("2. CPU-GPU Data Sharing:\n");
    
    // Reset data
    for (int i = 0; i < n; i++) {
        um_data[i] = 1.0f;
    }
    
    // GPU processes first
    printf("   GPU processing (x2 + 1)...\n");
    unifiedMemoryKernel<<<blocksPerGrid, threadsPerBlock>>>(um_data, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // CPU processes next
    printf("   CPU processing (+10)...\n");
    cpuProcess(um_data, 100);  // Just first 100
    
    printf("   First 5 elements: ");
    for (int i = 0; i < 5; i++) {
        printf("%.1f ", um_data[i]);
    }
    printf("(expected: 13.0 = (1*2+1)+10)\n\n");
    
    // ========================================================================
    // MEMORY PREFETCHING
    // Hint to CUDA where data will be accessed
    // ========================================================================
    printf("3. Memory Prefetching:\n");
    
    float *um_input, *um_output;
    CUDA_CHECK(cudaMallocManaged(&um_input, size));
    CUDA_CHECK(cudaMallocManaged(&um_output, size));
    
    for (int i = 0; i < n; i++) {
        um_input[i] = i * 0.001f;
    }
    
    // Prefetch to GPU before kernel
    int deviceId = 0;
    CUDA_CHECK(cudaMemPrefetchAsync(um_input, size, deviceId));
    CUDA_CHECK(cudaMemPrefetchAsync(um_output, size, deviceId));
    
    printf("   Running with prefetch to GPU...\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    prefetchedKernel<<<blocksPerGrid, threadsPerBlock>>>(um_input, um_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float prefetchedTime;
    cudaEventElapsedTime(&prefetchedTime, start, stop);
    printf("   Time with prefetch: %.3f ms\n", prefetchedTime);
    
    // Without prefetch (let UM handle it)
    for (int i = 0; i < n; i++) {
        um_output[i] = 0;
    }
    
    printf("   Running without prefetch...\n");
    cudaEventRecord(start);
    prefetchedKernel<<<blocksPerGrid, threadsPerBlock>>>(um_input, um_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float noPrefetchTime;
    cudaEventElapsedTime(&noPrefetchTime, start, stop);
    printf("   Time without prefetch: %.3f ms\n\n", noPrefetchTime);
    
    // ========================================================================
    // MEMORY ADVICE
    // Tell CUDA about access patterns
    // ========================================================================
    printf("4. Memory Advice (Access Pattern Hints):\n");
    
    // Advise that data is mostly read by GPU
    CUDA_CHECK(cudaMemAdvise(um_input, size, cudaMemAdviseSetReadMostly, deviceId));
    
    // Advise that data will be accessed by CPU
    CUDA_CHECK(cudaMemAdvise(um_output, size, cudaMemAdviseSetPreferredLocation, 
                             cudaCpuDeviceId));
    
    printf("   Set ReadMostly advice on input\n");
    printf("   Set PreferredLocation (CPU) on output\n\n");
    
    // Cleanup
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(um_data);
    cudaFree(um_input);
    cudaFree(um_output);
    
    printf("=== Unified Memory Summary ===\n");
    printf("| Feature              | Traditional CUDA    | Unified Memory      |\n");
    printf("|----------------------|---------------------|---------------------|\n");
    printf("| Allocation           | cudaMalloc            | cudaMallocManaged |\n");
    printf("| Data transfer        | cudaMemcpy            | Automatic         |\n");
    printf("| CPU access           | Requires copy back    | Direct access     |\n");
    printf("| Optimization         | Manual                | Prefetch/Advice   |\n");
    printf("| Complexity           | Higher                | Lower             |\n");
    printf("\n=== Best Practices ===\n");
    printf("1. Use cudaMallocManaged for simplicity\n");
    printf("2. Use cudaMemPrefetchAsync for known access patterns\n");
    printf("3. Use cudaMemAdvise for read-mostly or preferred location\n");
    printf("4. Profile with nvprof/nsys to see page faults\n");
    printf("5. For performance-critical code, explicit memcpy may be faster\n");
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Unified Memory:
//    - Single pointer accessible by CPU and GPU
//    - Automatic data migration (paging)
//    - Simplifies code, may have overhead
//
// 2. cudaMallocManaged:
//    - Allocates managed memory
//    - Data migrates on-demand (page faults)
//    - Synchronized at kernel launch / cudaDeviceSynchronize
//
// 3. Optimization:
//    - cudaMemPrefetchAsync: Proactively move data
//    - cudaMemAdvise: Hint about access patterns
//    - cudaMemAdviseSetReadMostly: Data rarely modified
//    - cudaMemAdviseSetPreferredLocation: Where data lives
//
// 4. When to Use:
//    - Good: Irregular access patterns, simpler code
//    - Bad: Performance-critical, predictable patterns
//
// EXERCISES:
// 1. Compare performance: UM vs explicit cudaMemcpy
// 2. Implement a linked list traversal on GPU with UM
// 3. Use cudaMemAdvise for your workload
// 4. Profile page faults with nvprof
// 5. Research: What is oversubscription with UM?
// ============================================================================
