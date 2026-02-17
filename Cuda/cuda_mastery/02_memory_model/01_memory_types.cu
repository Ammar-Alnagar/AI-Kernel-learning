// ============================================================================
// Lesson 2.1: Memory Types - Understanding CUDA Memory Hierarchy
// ============================================================================
// Concepts Covered:
//   - Global memory (device memory)
//   - Constant memory
//   - Local memory (spill to DRAM)
//   - Memory characteristics and use cases
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Constant memory declaration (max 64KB total)
// Stored in special constant cache, optimized for broadcast
__constant__ float d_constantValue;
__constant__ float d_constantArray[256];

// ============================================================================
// Global Memory Access
// - High latency (~400-800 cycles)
// - High bandwidth
// - Cached in L1/L2 (compute capability dependent)
// ============================================================================
__global__ void globalMemoryDemo(float *input, float *output, int n, float multiplier) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Read from global memory, write to global memory
        float value = input[idx];
        output[idx] = value * multiplier;
    }
}

// ============================================================================
// Constant Memory Access
// - Cached in constant cache
// - Optimized for broadcast (all threads read same address)
// - Slow if threads read different addresses (serialized)
// ============================================================================
__global__ void constantMemoryDemo(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // All threads read the same constant - optimized broadcast
        output[idx] = input[idx] * d_constantValue;
    }
}

// ============================================================================
// Constant Array Access
// - Fast if all threads read same index
// - Slow if threads read different indices (serialized)
// ============================================================================
__global__ void constantArrayDemo(float *input, float *output, int n, int arrayIdx) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Reading same index from constant array - fast
        output[idx] = input[idx] * d_constantArray[arrayIdx];
    }
}

// ============================================================================
// Local Memory Example (Register Spill)
// - Used when registers are exhausted
// - Actually stored in global memory (slow!)
// - Compiler decides when to spill
// ============================================================================
__global__ void localMemoryDemo(float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Large local array may cause register spill to local memory
        float largeArray[64];  // May spill to local memory
        
        for (int i = 0; i < 64; i++) {
            largeArray[i] = idx * i;
        }
        
        float sum = 0;
        for (int i = 0; i < 64; i++) {
            sum += largeArray[i];
        }
        
        output[idx] = sum;
    }
}

// ============================================================================
// Memory Access Pattern Comparison
// ============================================================================
__global__ void compareMemoryTypes(float *globalData, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Global memory read
        float globalVal = globalData[idx];
        
        // Constant memory read (same for all threads)
        float constVal = d_constantValue;
        
        output[idx] = globalVal + constVal;
    }
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    // Host arrays
    float *h_input = (float *)malloc(size);
    float *h_output = (float *)malloc(size);
    
    // Initialize
    for (int i = 0; i < n; i++) {
        h_input[i] = i * 0.5f;
    }
    
    // Device arrays
    float *d_input, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, size));
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    // Copy to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input, size, cudaMemcpyHostToDevice));
    
    // Copy constant value to constant memory
    float constantVal = 100.0f;
    CUDA_CHECK(cudaMemcpyToSymbol(d_constantValue, &constantVal, sizeof(float)));
    
    // Copy constant array
    float constArray[256];
    for (int i = 0; i < 256; i++) {
        constArray[i] = i * 1.0f;
    }
    CUDA_CHECK(cudaMemcpyToSymbol(d_constantArray, constArray, sizeof(constArray)));
    
    // Execution config
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("=== Global Memory Demo ===\n");
    globalMemoryDemo<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, 2.0f);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", h_output[i]);
    printf("\n");
    
    printf("\n=== Constant Memory Demo ===\n");
    constantMemoryDemo<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", h_output[i]);
    printf("\n");
    
    printf("\n=== Constant Array Demo ===\n");
    constantArrayDemo<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, n, 5);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", h_output[i]);
    printf("\n");
    
    printf("\n=== Local Memory Demo ===\n");
    localMemoryDemo<<<blocksPerGrid, threadsPerBlock>>>(d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("First 5 results: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", h_output[i]);
    printf("\n");
    
    // Cleanup
    cudaFree(d_input);
    cudaFree(d_output);
    free(h_input);
    free(h_output);
    
    printf("\n=== Memory Type Summary ===\n");
    printf("| Type       | Location | Cached | Latency    | Best For           |\n");
    printf("|------------|----------|--------|------------|--------------------|\n");
    printf("| Global     | DRAM     | L1/L2  | High       | Large datasets     |\n");
    printf("| Constant   | DRAM     | Const  | Low*       | Read-only uniform  |\n");
    printf("| Local      | DRAM     | L1/L2  | High       | Register overflow  |\n");
    printf("| Shared     | On-chip  | Yes    | Very Low   | Thread cooperation |\n");
    printf("| Registers  | On-chip  | N/A    | Lowest     | Private variables  |\n");
    printf("\n*Low latency only when all threads read same address\n");
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. Global Memory:
//    - Main device memory (GPU DRAM)
//    - Accessible by all threads
//    - High latency, high bandwidth
//    - Use cudaMalloc/cudaMemcpy
//
// 2. Constant Memory:
//    - 64KB limit
//    - Cached in constant cache
//    - Optimized for broadcast pattern
//    - Use cudaMemcpyToSymbol
//
// 3. Local Memory:
//    - Not a physical memory type
//    - Register spill space in DRAM
//    - Should be avoided (slow!)
//    - Automatic, not explicitly managed
//
// EXERCISES:
// 1. Benchmark global vs constant memory for broadcast pattern
// 2. What happens when constant memory is accessed with different indices?
// 3. Try to force register spill by using more local variables
// 4. Research: How to check if your kernel spills to local memory?
// ============================================================================
