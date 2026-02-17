// ============================================================================
// Lesson 1.1: Hello CUDA - Your First GPU Program
// ============================================================================
// Concepts Covered:
//   - Kernel definition (__global__)
//   - Kernel launch syntax <<<grid, block>>>
//   - Thread indexing (threadIdx.x)
//   - Basic GPU parallelism
// ============================================================================

#include <stdio.h>

// Kernel: Function that runs on GPU
// __global__ means it's called from host and executes on device
__global__ void helloCUDA(int threadId) {
    // Each thread executes this function independently
    printf("Hello from thread %d on GPU!\n", threadId);
}

int main() {
    int numThreads = 8;  // Number of parallel threads
    
    printf("Launching %d threads on GPU...\n\n", numThreads);
    
    // Kernel launch syntax: kernelName<<<gridSize, blockSize>>>(args)
    // Here: 1 block with numThreads threads
    helloCUDA<<<1, numThreads>>>(threadIdx.x);
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();
    
    printf("\nAll threads completed!\n");
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. __global__: Function qualifier indicating a kernel (GPU function)
// 2. <<<grid, block>>>: Execution configuration
//    - grid: Number of blocks
//    - block: Threads per block
// 3. threadIdx.x: Built-in variable giving thread's index within block
// 4. cudaDeviceSynchronize(): Host waits for GPU to complete
//
// EXERCISES:
// 1. Change numThreads to 16, 32, 64, 128, 256, 512, 1024
// 2. What's the maximum threads per block on your GPU?
// 3. Try launching with multiple blocks: helloCUDA<<<2, 4>>>()
//    - How does threadIdx.x behave now?
// ============================================================================
