// ============================================================================
// Lesson 4.1: Synchronization and Atomics - Coordinating Threads
// ============================================================================
// Concepts Covered:
//   - __syncthreads() barrier
//   - Atomic operations (add, min, max, CAS)
//   - Race conditions and prevention
//   - Warp-level primitives (shfl)
// ============================================================================

#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error: %s\n", cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// SYNCHRONIZATION WITH __syncthreads()
// Block-wide barrier - all threads must reach it
// ============================================================================

// WRONG: Conditional __syncthreads() - causes undefined behavior!
__global__ void wrongSync(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        data[idx] = 1.0f;
        __syncthreads();  // DANGER: Not all threads reach here!
    }
}

// CORRECT: All threads must execute __syncthreads()
__global__ void correctSync(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // All threads participate in synchronization
    if (idx < n) {
        data[idx] = 1.0f;
    }
    __syncthreads();  // SAFE: All threads reach this
    
    if (idx < n) {
        // Now safe to read other threads' data
        data[idx] += 1.0f;
    }
}

// ============================================================================
// ATOMIC OPERATIONS
// Thread-safe operations on global/shared memory
// ============================================================================

// Atomic add - prevents race condition
__global__ void atomicAddKernel(float *data, float *sum, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Atomically add to shared sum
        // No race condition - hardware ensures atomicity
        atomicAdd(sum, data[idx]);
    }
}

// Atomic compare-and-swap (CAS) - building block for locks
__global__ void atomicCASKernel(int *lock, int *counter, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Simple lock using CAS
        int expected = 0;
        while (atomicCAS(lock, expected, 1) != expected) {
            expected = 0;  // Reset for next attempt
        }
        
        // Critical section - only one thread at a time
        (*counter)++;
        
        // Release lock
        atomicExch(lock, 0);
    }
}

// Atomic min/max
__global__ void atomicMinMaxKernel(float *data, float *minVal, float *maxVal, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = data[idx];
        
        // Atomically update minimum
        float oldMin = *minVal;
        while (val < oldMin) {
            float prev = atomicMin(minVal, __float2int_rn(val));
            oldMin = __int2float_rn(prev);
        }
        
        // Atomically update maximum
        float oldMax = *maxVal;
        while (val > oldMax) {
            float prev = atomicMax(maxVal, __float2int_rn(val));
            oldMax = __int2float_rn(prev);
        }
    }
}

// ============================================================================
// WARP-LEVEL PRIMITIVES (Shuffle Operations)
// Faster than shared memory for intra-warp communication
// ============================================================================

// Warp shuffle - exchange data between threads in a warp
__global__ void warpShuffleDemo(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // Shuffle: get value from lane (thread) offset by 1
        // laneId = threadIdx.x % 32 (within warp)
        int laneId = threadIdx.x % 32;
        
        // Get value from next thread in warp (with wraparound)
        float neighborVal = __shfl_sync(0xFFFFFFFF, val, (laneId + 1) % 32);
        
        output[idx] = val + neighborVal;
    }
}

// Warp shuffle down - get value from thread with higher lane ID
__global__ void warpShuffleDownDemo(float *input, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float val = input[idx];
        
        // Get value from thread 2 lanes ahead
        float valPlus2 = __shfl_down_sync(0xFFFFFFFF, val, 2);
        
        output[idx] = val + valPlus2;
    }
}

// Warp-level reduction (much faster than shared memory reduction)
__global__ void warpReduceKernel(float *data, float *blockSums, int n) {
    __shared__ float sharedData[1024];
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // Load data
    sharedData[tid] = (idx < n) ? data[idx] : 0.0f;
    __syncthreads();
    
    // Shared memory reduction for first 32 elements
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1) {
        if (tid < stride) {
            sharedData[tid] += sharedData[tid + stride];
        }
        __syncthreads();
    }
    
    // Warp-level reduction (no sync needed - implicit warp sync)
    if (tid < 32) {
        float val = sharedData[tid];
        val += __shfl_down_sync(0xFFFFFFFF, val, 16);
        val += __shfl_down_sync(0xFFFFFFFF, val, 8);
        val += __shfl_down_sync(0xFFFFFFFF, val, 4);
        val += __shfl_down_sync(0xFFFFFFFF, val, 2);
        val += __shfl_down_sync(0xFFFFFFFF, val, 1);
        
        if (tid == 0) {
            blockSums[blockIdx.x] = val;
        }
    }
}

int main() {
    int n = 10000;
    size_t size = n * sizeof(float);
    
    float *h_data = (float *)malloc(size);
    for (int i = 0; i < n; i++) {
        h_data[i] = 1.0f;
    }
    
    float *d_data, *d_sum;
    CUDA_CHECK(cudaMalloc(&d_data, size));
    CUDA_CHECK(cudaMalloc(&d_sum, sizeof(float)));
    
    CUDA_CHECK(cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice));
    
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("=== Synchronization and Atomics Demo ===\n\n");
    
    // Atomic sum
    float h_sum = 0.0f;
    CUDA_CHECK(cudaMemcpy(d_sum, &h_sum, sizeof(float), cudaMemcpyHostToDevice));
    
    printf("Atomic Addition Test:\n");
    printf("  Input: %d elements, each = 1.0\n", n);
    
    atomicAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_sum, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(&h_sum, d_sum, sizeof(float), cudaMemcpyDeviceToHost));
    printf("  Atomic sum result: %.0f (expected: %d)\n\n", h_sum, n);
    
    // Warp shuffle demo
    float *d_output;
    CUDA_CHECK(cudaMalloc(&d_output, size));
    
    printf("Warp Shuffle Test:\n");
    warpShuffleDemo<<<blocksPerGrid, threadsPerBlock>>>(d_data, d_output, n);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    float *h_output = (float *)malloc(size);
    CUDA_CHECK(cudaMemcpy(h_output, d_output, size, cudaMemcpyDeviceToHost));
    printf("  First 5 results: ");
    for (int i = 0; i < 5; i++) printf("%.1f ", h_output[i]);
    printf("\n  (Each = itself + neighbor from shuffle)\n\n");
    
    // Cleanup
    cudaFree(d_data);
    cudaFree(d_sum);
    cudaFree(d_output);
    free(h_data);
    free(h_output);
    
    printf("=== Atomic Operations Summary ===\n");
    printf("| Operation    | Description                    | Latency    |\n");
    printf("|--------------|--------------------------------|------------|\n");
    printf("| atomicAdd    | Atomic addition                | ~500 cycles|\n");
    printf("| atomicSub    | Atomic subtraction             | ~500 cycles|\n");
    printf("| atomicExch   | Atomic exchange                | ~500 cycles|\n");
    printf("| atomicMin/Max| Atomic min/max (int)           | ~500 cycles|\n");
    printf("| atomicCAS    | Compare-and-swap               | ~500 cycles|\n");
    printf("| __shfl_sync  | Warp shuffle (intra-warp)      | ~10 cycles |\n");
    printf("\n=== Synchronization Rules ===\n");
    printf("1. __syncthreads() must be reached by ALL threads in block\n");
    printf("2. Never put __syncthreads() in conditionals\n");
    printf("3. Use atomics for cross-block synchronization\n");
    printf("4. Warp shuffles are faster than shared memory (intra-warp only)\n");
    
    return 0;
}

// ============================================================================
// KEY CONCEPTS:
// ============================================================================
// 1. __syncthreads():
//    - Block-wide barrier
//    - ALL threads must execute it (no conditionals!)
//    - Ensures memory visibility across block
//
// 2. Atomic Operations:
//    - Thread-safe read-modify-write
//    - Prevent race conditions
//    - Slower than regular operations (~500 cycles)
//    - Use sparingly!
//
// 3. Warp-Level Primitives:
//    - __shfl_sync: Exchange data within warp
//    - __shfl_down_sync: Get data from higher lane
//    - __shfl_up_sync: Get data from lower lane
//    - Much faster than shared memory (~10 cycles)
//    - Only works within a warp (32 threads)
//
// EXERCISES:
// 1. Implement a lock using atomicCAS
// 2. Create a warp-level broadcast using shfl
// 3. Compare performance: atomic vs shared memory reduction
// 4. Research: What is a memory fence (__threadfence)?
// 5. Implement a simple spinlock
// ============================================================================
