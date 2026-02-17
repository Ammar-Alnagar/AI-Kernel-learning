// ============================================================================
// Lesson 2.2: Memory Coalescing - Optimizing Global Memory Access
// ============================================================================
// Concepts Covered:
//   - Coalesced vs uncoalesced memory access
//   - Memory access patterns
//   - Performance impact of access patterns
//   - Strided access patterns
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// COALESCED ACCESS (Optimal)
// Consecutive threads access consecutive memory locations
// Results in single memory transaction
// ============================================================================
__global__ void coalescedRead(const float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Threads 0,1,2,3... access indices 0,1,2,3...
        // This is COALESCED - optimal!
        output[idx] = input[idx] * 2.0f;
    }
}

// ============================================================================
// UNCOALESCED ACCESS (Poor Performance)
// Consecutive threads access far-apart memory locations
// Results in multiple memory transactions
// ============================================================================
__global__ void uncoalescedRead(const float *input, float *output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Thread 0 accesses index 0
        // Thread 1 accesses index stride
        // Thread 2 accesses index 2*stride
        // This is UNCOALESCED if stride is large!
        output[idx] = input[idx * stride] * 2.0f;
    }
}

// ============================================================================
// TRANSPOSE - Classic Uncoalesced Pattern
// Reading rows, writing columns (or vice versa)
// ============================================================================
__global__ void transposeNaive(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int inputIdx = y * width + x;      // Row-major read
        int outputIdx = x * height + y;    // Column-major write
        
        output[outputIdx] = input[inputIdx];
    }
}

// ============================================================================
// COALESCED TRANSPOSE (Using Shared Memory)
// See shared_memory lessons for optimized version
// This is a simple strided approach
// ============================================================================
__global__ void transposeCoalesced(float *input, float *output, int width, int height) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x < width && y < height) {
        int inputIdx = y * width + x;
        int outputIdx = x * height + y;
        
        // Read coalesced, write uncoalesced OR
        // Read uncoalesced, write coalesced (pick one)
        output[outputIdx] = input[inputIdx];
    }
}

// ============================================================================
// STRUCT OF ARRAYS vs ARRAY OF STRUCTS
// SoA is better for coalescing when threads access same field
// ============================================================================
struct Point3D {
    float x, y, z;
};

// Array of Structures (AoS) - Poor coalescing for single component
__global__ void processAoS_XOnly(Point3D *points, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Only accessing .x component
        // Memory layout: x0,y0,z0, x1,y1,z1, x2,y2,z2...
        // Threads access: x0, x1, x2... (strided by 3 floats!)
        output[idx] = points[idx].x * 2.0f;
    }
}

// Structure of Arrays (SoA) - Good coalescing
__global__ void processSoA_XOnly(float *xCoords, float *yCoords, float *zCoords,
                                  float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // All x values are contiguous
        // Memory layout: x0,x1,x2,x3... y0,y1,y2,y3... z0,z1,z2,z3...
        // Threads access: x0, x1, x2... (contiguous!)
        output[idx] = xCoords[idx] * 2.0f;
    }
}

// Timing helper
void runKernel(void (*kernel)(const float*, float*, int), 
               float *d_in, float *d_out, int n, 
               int blocks, int threads, const char *name) {
    // Warmup
    kernel<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();
    
    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start);
    for (int i = 0; i < 100; i++) {
        kernel<<<blocks, threads>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    
    float elapsed;
    cudaEventElapsedTime(&elapsed, start);
    
    printf("%-20s: %.3f ms (100 iterations)\n", name, elapsed);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

int main() {
    int n = 1 << 20;  // 1M elements = 4MB
    size_t size = n * sizeof(float);
    
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    
    for (int i = 0; i < n; i++) {
        h_input[i] = i * 1.0f;
    }
    
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Array size: %d elements (%.2f MB)\n", n, size / (1024.0 * 1024.0));
    printf("Configuration: %d blocks, %d threads/block\n\n", blocksPerGrid, threadsPerBlock);
    
    printf("=== Memory Access Pattern Comparison ===\n\n");
    
    // Coalesced access
    runKernel(coalescedRead, d_input, d_output, n, blocksPerGrid, threadsPerBlock,
              "Coalesced Read");
    
    // Uncoalesced access with different strides
    printf("\nUncoalesced with stride 2: ");
    uncoalescedRead<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n/2, 2);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    printf("\n=== Coalescing Principles ===\n");
    printf("1. Consecutive threads -> consecutive addresses\n");
    printf("2. Avoid strided access when possible\n");
    printf("3. Use SoA instead of AoS for SIMD patterns\n");
    printf("4. Transpose requires shared memory optimization\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Memory Coalescing:
//    - GPU combines consecutive thread accesses into single transaction
//    - Requires consecutive threads accessing consecutive addresses
//    - Can improve bandwidth by 2x-16x!
//
// 2. Access Patterns:
//    - Coalesced: thread i accesses address i
//    - Strided: thread i accesses address i * stride
//    - Random: worst case, no pattern
//
// 3. AoS vs SoA:
//    - AoS: struct {x,y,z} arr[n] - bad for component-wise
//    - SoA: float x[n], y[n], z[n] - good for component-wise
//
// EXERCISES:
// 1. Add proper timing comparison between coalesced/uncoalesced
// 2. Implement SoA version and compare with AoS
// 3. Research: What is a memory transaction size on your GPU?
// 4. Use nvprof/nsys to measure memory throughput
// 5. Implement optimized transpose using shared memory
// ============================================================================
