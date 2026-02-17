// ============================================================================
// Lesson 6.1: CUDA Streams - Concurrent Execution
// ============================================================================
// Concepts Covered:
//   - Default vs non-default streams
//   - Concurrent kernel execution
//   - Async memory transfers
//   - Stream synchronization
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

// Simple kernel that takes time
__global__ void vectorAddKernel(float *A, float *B, float *C, int n, int iterations) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0;
        // Artificial work to make kernel take time
        for (int i = 0; i < iterations; i++) {
            sum += A[idx] * B[idx];
        }
        C[idx] = sum;
    }
}

// Kernel with configurable duration
__global__ void timedKernel(float *output, int n, int workAmount) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float sum = 0;
        for (int i = 0; i < workAmount; i++) {
            sum += i * 0.001f;
        }
        output[idx] = sum;
    }
}

int main() {
    int n = 1 << 20;  // 1M elements
    size_t size = n * sizeof(float);
    
    printf("=== CUDA Streams Demo ===\n\n");
    
    // Device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("Device: %s\n", prop.name);
    printf("Concurrent kernels: %s\n", 
           prop.concurrentKernels ? "Yes" : "No");
    printf("Async engine count: %d\n\n", prop.asyncEngineCount);
    
    // Host arrays
    float *h_A = (float *)malloc(size);
    float *h_B = (float *)malloc(size);
    float *h_C = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }
    
    // Device arrays
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc(&d_A, size));
    CUDA_CHECK(cudaMalloc(&d_B, size));
    CUDA_CHECK(cudaMalloc(&d_C, size));
    
    // Create streams
    const int NUM_STREAMS = 4;
    cudaStream_t streams[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamCreate(&streams[i]));
    }
    
    // Execution config
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    int workAmount = 1000;  // Adjust for timing
    
    printf("Configuration:\n");
    printf("  Array size: %d elements\n", n);
    printf("  Streams: %d\n", blocksPerGrid);
    printf("  Blocks per grid: %d\n\n", blocksPerGrid);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // ========================================================================
    // SEQUENTIAL EXECUTION (Default Stream)
    // ========================================================================
    printf("1. Sequential Execution (4 kernels in default stream):\n");
    
    cudaEventRecord(start);
    
    // Launch 4 kernels sequentially
    for (int i = 0; i < 4; i++) {
        vectorAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n, workAmount);
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float sequentialTime;
    cudaEventElapsedTime(&sequentialTime, start, stop);
    printf("   Time: %.3f ms\n\n", sequentialTime);
    
    // ========================================================================
    // CONCURRENT EXECUTION (Multiple Streams)
    // ========================================================================
    printf("2. Concurrent Execution (4 kernels in 4 streams):\n");
    
    // Split work across streams
    int elementsPerStream = n / NUM_STREAMS;
    size_t streamSize = elementsPerStream * sizeof(float);
    
    cudaEventRecord(start);
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        int streamOffset = i * elementsPerStream;
        
        // Launch kernel in stream with offset pointers
        vectorAddKernel<<<blocksPerGrid / NUM_STREAMS, threadsPerBlock, 0, streams[i]>>>(
            d_A + streamOffset, 
            d_B + streamOffset, 
            d_C + streamOffset, 
            elementsPerStream, 
            workAmount
        );
    }
    
    // Wait for all streams
    for (int i = 0; i < NUM_STREAMS; i++) {
        CUDA_CHECK(cudaStreamSynchronize(streams[i]));
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float concurrentTime;
    cudaEventElapsedTime(&concurrentTime, start, stop);
    printf("   Time: %.3f ms\n", concurrentTime);
    printf("   Speedup: %.2fx\n\n", sequentialTime / concurrentTime);
    
    // ========================================================================
    // ASYNC MEMORY TRANSFER WITH STREAMS
    // ========================================================================
    printf("3. Async Memory Transfer Demo:\n");
    
    // Pinned (page-locked) memory for async transfers
    float *h_A_pinned, *h_C_pinned;
    CUDA_CHECK(cudaMallocHost(&h_A_pinned, size));
    CUDA_CHECK(cudaMallocHost(&h_C_pinned, size));
    
    for (int i = 0; i < n; i++) {
        h_A_pinned[i] = 1.0f;
    }
    
    cudaEventRecord(start);
    
    // Overlap transfer and compute
    for (int i = 0; i < NUM_STREAMS; i++) {
        int streamOffset = i * elementsPerStream;
        
        // Async H2D transfer
        cudaMemcpyAsync(h_A_pinned + streamOffset, 
                       d_A + streamOffset, 
                       streamSize, 
                       cudaMemcpyDeviceToHost, 
                       streams[i]);
        
        // Kernel in same stream
        vectorAddKernel<<<blocksPerGrid / NUM_STREAMS, threadsPerBlock, 0, streams[i]>>>(
            d_A + streamOffset, d_B + streamOffset, d_C + streamOffset,
            elementsPerStream, workAmount
        );
    }
    
    CUDA_CHECK(cudaDeviceSynchronize());
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float asyncTime;
    cudaEventElapsedTime(&asyncTime, start, stop);
    printf("   Time with async transfers: %.3f ms\n\n", asyncTime);
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        cudaStreamDestroy(streams[i]);
    }
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFreeHost(h_A_pinned);
    cudaFreeHost(h_C_pinned);
    free(h_A);
    free(h_B);
    free(h_C);
    
    printf("=== Streams Summary ===\n");
    printf("| Feature              | Default Stream | Non-Default Stream |\n");
    printf("|----------------------|----------------|--------------------|\n");
    printf("| Synchronization      | Implicit       | Explicit           |\n");
    printf("| Concurrent kernels   | No             | Yes                |\n");
    printf("| Async memcpy         | No             | Yes (with pinned)  |\n");
    printf("| Priority             | Normal         | Configurable       |\n");
    printf("\n=== Stream Best Practices ===\n");
    printf("1. Use cudaStreamCreateWithFlags(cudaStreamNonBlocking) for more concurrency\n");
    printf("2. Pinned memory required for async cudaMemcpyAsync\n");
    printf("3. Use cudaStreamWaitEvent() for inter-stream synchronization\n");
    printf("4. Query device properties for concurrent kernel support\n");
    printf("5. Use Nsight Systems to visualize stream concurrency\n");
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. CUDA Streams:
//    - Queue of operations (kernels, transfers)
//    - Default stream (0): synchronous, implicit sync
//    - Non-default streams: asynchronous, can run concurrently
//
// 2. Concurrency:
//    - Multiple kernels from different streams can run together
//    - Requires GPU support (concurrentKernels = true)
//    - Limited by GPU resources (SMs, memory bandwidth)
//
// 3. Async Memory Transfers:
//    - cudaMemcpyAsync with non-default stream
//    - Requires pinned (page-locked) host memory
//    - Can overlap with kernel execution
//
// 4. Stream Synchronization:
//    - cudaStreamSynchronize(): Wait for specific stream
//    - cudaDeviceSynchronize(): Wait for all streams
//    - cudaStreamWaitEvent(): Stream waits on event
//
// EXERCISES:
// 1. Implement producer-consumer with two streams
// 2. Use cudaStreamWaitEvent() to synchronize streams
// 3. Try cudaStreamCreateWithPriority() for QoS
// 4. Profile with Nsight Systems to see concurrency
// 5. Research: What is the Hyper-Q feature?
// ============================================================================
