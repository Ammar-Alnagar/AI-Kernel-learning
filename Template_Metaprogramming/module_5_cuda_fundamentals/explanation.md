# Module 5: CUDA and GPU Programming Fundamentals

## Overview
This module provides essential knowledge of GPU programming concepts necessary for understanding and working with CUTLASS. We'll cover GPU architecture, CUDA programming model, memory hierarchies, and performance considerations.

## Learning Objectives
By the end of this module, students will be able to:
- Understand GPU architecture basics and how it differs from CPU architecture
- Program with the CUDA execution model
- Work with different GPU memory types (global, shared, registers)
- Organize threads using blocks and grids
- Optimize memory access patterns for coalescing
- Understand occupancy and performance factors

## Topic 1: GPU Architecture Basics

### GPU vs CPU Architecture
GPUs are designed for parallel processing with thousands of cores optimized for throughput rather than latency.

```cuda
// CPU vs GPU characteristics
/*
CPU:
- Few powerful cores (4-64)
- High clock speed (3-5 GHz)
- Complex control logic
- Large caches
- Optimized for sequential code

GPU:
- Thousands of simpler cores (1000s)
- Lower clock speed (1-2 GHz)
- Simpler control logic per core
- Smaller caches per core
- Optimized for parallel execution
*/
```

### SIMD vs SIMT
- CPUs typically use SIMD (Single Instruction, Multiple Data)
- GPUs use SIMT (Single Instruction, Multiple Thread)
- In SIMT, many threads execute the same instruction on different data

```cuda
// SIMT example - all threads in a warp execute the same instruction
__global__ void simt_example(float* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // All threads in warp execute this together
        data[tid] = data[tid] * 2.0f;
    }
}
```

## Topic 2: CUDA Programming Model

### Basic CUDA Program Structure
```cuda
#include <cuda_runtime.h>
#include <iostream>

// Kernel function - runs on GPU
__global__ void vector_add(float* a, float* b, float* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(float);
    
    // Host arrays
    float *h_a, *h_b, *h_c;
    h_a = new float[N];
    h_b = new float[N];
    h_c = new float[N];
    
    // Device arrays
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);
    
    // Initialize host arrays
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i * 2;
    }
    
    // Copy data to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);
    
    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    
    // Wait for kernel to complete
    cudaDeviceSynchronize();
    
    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);
    
    // Cleanup
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    
    return 0;
}
```

### Function Type Qualifiers
```cuda
// Runs on host, called from host
void host_function() { }

// Runs on device, called from device
__device__ void device_function() { }

// Runs on device, called from host
__global__ void kernel_function() { }

// Runs on both host and device
__host__ __device__ float utility_function(float x) {
    return x * x;
}
```

## Topic 3: Thread Organization

### Grid, Block, and Thread Hierarchy
```cuda
// Understanding thread indexing
__global__ void example_kernel() {
    // Global thread ID
    int global_id = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Local thread ID within block
    int local_id = threadIdx.x;
    
    // Block ID
    int block_id = blockIdx.x;
    
    // Total threads in grid
    int total_threads = gridDim.x * blockDim.x;
    
    // Use these IDs to determine work distribution
    if (global_id < total_elements) {
        // Process element at global_id
    }
}
```

### 2D and 3D Grid/Block Organization
```cuda
// 2D grid example - useful for matrix operations
__global__ void matrix_operation(float* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;
        matrix[idx] = matrix[idx] * 2.0f;
    }
}

// Launch with 2D configuration
dim3 blockSize(16, 16);  // 16x16 = 256 threads per block
dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
              (height + blockSize.y - 1) / blockSize.y);

matrix_operation<<<gridSize, blockSize>>>(d_matrix, height, width);
```

### Warps and Warp Execution
```cuda
// Understanding warps (groups of 32 threads)
__global__ void warp_example(int* data, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Threads 0-31 form warp 0, 32-63 form warp 1, etc.
    int warp_id = threadIdx.x / 32;
    int lane_id = threadIdx.x % 32;  // Position within warp (0-31)
    
    if (tid < n) {
        // All threads in warp execute this instruction together
        data[tid] = data[tid] + warp_id;
    }
    
    // Warp-level primitives (CUDA 9.0+)
    if (lane_id < 16) {
        // Cooperative operations within warp
        int partner = lane_id + 16;
        if (partner < 32) {
            // Swap values with partner in warp
            int temp = data[tid];
            data[tid] = data[blockIdx.x * blockDim.x + partner];
        }
    }
}
```

## Topic 4: Memory Hierarchies

### Global Memory
```cuda
// Global memory - largest, slowest, accessible by all threads
__global__ void global_memory_example(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Global memory access - relatively slow
        float value = input[idx];
        output[idx] = value * 2.0f;
    }
}
```

### Shared Memory
```cuda
// Shared memory - faster, shared among threads in a block
__global__ void shared_memory_example(float* input, float* output, int n) {
    extern __shared__ float shared_data[];
    
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load data into shared memory
    if (gid < n) {
        shared_data[tid] = input[gid];
    }
    __syncthreads();  // Ensure all threads have loaded
    
    // Process data in shared memory
    if (tid > 0 && gid < n) {
        shared_data[tid] += shared_data[tid - 1];
    }
    __syncthreads();
    
    // Write back to global memory
    if (gid < n) {
        output[gid] = shared_data[tid];
    }
}

// Launch with shared memory
int shared_mem_size = 256 * sizeof(float);  // 256 threads per block
vector_add<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_input, d_output, N);
```

### Constant Memory
```cuda
// Constant memory - cached, read-only, accessible by all threads
__constant__ float coefficients[256];

__global__ void constant_memory_example(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n && idx < 256) {
        // Access constant memory - cached and fast
        output[idx] = input[idx] * coefficients[idx];
    }
}

// Copy to constant memory from host
float h_coefficients[256];
// ... initialize h_coefficients ...
cudaMemcpyToSymbol(coefficients, h_coefficients, sizeof(h_coefficients));
```

### Texture Memory
```cuda
// Texture memory - cached, optimized for spatial locality
texture<float, 1, cudaReadModeElementType> tex_input;

__global__ void texture_memory_example(float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Texture memory access with interpolation
        float value = tex1Dfetch(tex_input, idx);
        output[idx] = value * 2.0f;
    }
}

// Bind texture memory
cudaBindTexture(0, tex_input, d_input, size);
```

## Topic 5: Coalesced Memory Access

### Coalesced vs Uncoalesced Access
```cuda
// GOOD: Coalesced access - consecutive threads access consecutive memory
__global__ void coalesced_access(float* input, float* output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Threads 0,1,2,3... access memory addresses 0,1,2,3...
        output[idx] = input[idx] * 2.0f;
    }
}

// BAD: Uncoalesced access - threads access scattered memory
__global__ void uncoalesced_access(float* input, float* output, int n, int stride) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Threads 0,1,2,3... access memory addresses 0,stride,2*stride,3*stride...
        output[idx] = input[idx * stride] * 2.0f;
    }
}
```

### Matrix Memory Access Patterns
```cuda
// Row-major access (good for row-wise operations)
__global__ void row_major_access(float* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = row * cols + col;  // Consecutive columns accessed
        matrix[idx] = matrix[idx] + 1.0f;
    }
}

// Column-major access (good for column-wise operations)
__global__ void column_major_access(float* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows && col < cols) {
        int idx = col * rows + row;  // Consecutive rows accessed
        matrix[idx] = matrix[idx] + 1.0f;
    }
}
```

## Topic 6: Occupancy and Performance

### Occupancy Calculation
```cuda
// Calculate theoretical occupancy
void calculate_occupancy() {
    int device;
    cudaGetDevice(&device);
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);
    
    int blockSize = 256;
    int minGridSize, blockSizeOpt;
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSizeOpt, 
                                       vector_add, 0, 0);
    
    int maxActiveBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, 
                                                  vector_add, blockSize, 0);
    
    float occupancy = (maxActiveBlocks * blockSize / prop.warpSize) /
                      (float)(prop.maxThreadsPerMultiProcessor / prop.warpSize);
    
    printf("Theoretical occupancy: %.2f%%\n", occupancy * 100);
}
```

### Occupancy Optimization
```cuda
// Optimize for occupancy by adjusting block size
__global__ void optimized_kernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Balance between: 
    // - Enough threads per block to hide latency
    // - Not too many registers per block
    // - Not too much shared memory per block
    
    if (idx < n) {
        data[idx] = data[idx] * 2.0f + 1.0f;
    }
}

// Choose block size based on occupancy calculator
void launch_optimized() {
    int min_grid_size, block_size;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size, 
                                       optimized_kernel, 0, 0);
    
    int num_blocks = (N + block_size - 1) / block_size;
    optimized_kernel<<<num_blocks, block_size>>>(d_data, N);
}
```

## Topic 7: Memory Bandwidth Optimization

### Bank Conflicts in Shared Memory
```cuda
// BAD: Causes bank conflicts
__global__ void bank_conflict_example() {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    
    // If multiple threads access the same bank simultaneously
    // it causes serialization
    sdata[tid * 2] = tid;  // Threads 0,16,32... access same bank
}

// GOOD: Avoids bank conflicts
__global__ void no_bank_conflict_example() {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    
    // Access pattern avoids bank conflicts
    sdata[tid] = tid;
}
```

### Memory Padding to Avoid Conflicts
```cuda
// Pad shared memory to avoid bank conflicts
__global__ void padded_shared_memory(float* input, float* output, int n) {
    // Add one extra element per row to avoid bank conflicts
    extern __shared__ float sdata[][33];  // 32 + 1 to avoid conflicts
    
    int row = threadIdx.y;
    int col = threadIdx.x;
    
    if (row < 32 && col < 32) {
        int g_idx = blockIdx.y * 32 * gridDim.x * 32 + 
                    blockIdx.x * 32 + row * gridDim.x * 32 + col;
        if (g_idx < n) {
            sdata[row][col] = input[g_idx];
        }
    }
    __syncthreads();
    
    // Process data...
}
```

## Hands-on Exercises

### Exercise 1: Matrix Multiplication Kernel
Implement a basic matrix multiplication kernel optimizing for memory coalescing.

```cuda
// TODO: Implement matrix multiplication C = A * B
// Requirements:
// 1. Use proper indexing for coalesced memory access
// 2. Consider the memory access patterns for A, B, and C
// 3. Use appropriate grid/block dimensions
// 4. Test with different matrix sizes
```

### Exercise 2: Shared Memory Reduction
Implement a reduction operation using shared memory to minimize global memory accesses.

```cuda
// TODO: Implement parallel reduction using shared memory
// Requirements:
// 1. Use shared memory to store intermediate results
// 2. Implement proper synchronization with __syncthreads()
// 3. Handle cases where array size is not power of 2
// 4. Optimize for warp-level operations
```

### Exercise 3: Memory Access Pattern Analysis
Analyze and optimize a kernel for memory access patterns.

```cuda
// TODO: Given a kernel with poor memory access patterns,
// optimize it for better coalescing and performance
// Requirements:
// 1. Identify the problematic access patterns
// 2. Reorganize the data access for better coalescing
// 3. Measure performance improvement
```

## Solutions to Exercises

### Solution 1: Matrix Multiplication Kernel
```cuda
// Naive matrix multiplication - focus on coalesced access
__global__ void matrix_mult_naive(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}

// Tiled matrix multiplication for better memory access
__global__ void matrix_mult_tiled(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    const int TILE_SIZE = 16;
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    
    float sum = 0.0f;
    
    // Loop over tiles
    for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
        // Load tiles into shared memory
        if (row < M && t * TILE_SIZE + tx < K) {
            As[ty][tx] = A[row * K + t * TILE_SIZE + tx];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (col < N && t * TILE_SIZE + ty < K) {
            Bs[ty][tx] = B[(t * TILE_SIZE + ty) * N + col];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial sum
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// Launch configuration
void launch_matrix_mult() {
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x,
                  (M + blockSize.y - 1) / blockSize.y);
    
    matrix_mult_tiled<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
}
```

### Solution 2: Shared Memory Reduction
```cuda
// Parallel reduction using shared memory
template<typename T>
__device__ T warp_reduce_sum(T val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    }
    return val;
}

template<typename T>
__global__ void reduction_kernel(T* input, T* output, int n) {
    extern __shared__ T sdata[];
    
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    
    // Load data into shared memory
    T sum = 0;
    if (i < n) sum = input[i];
    if (i + blockDim.x < n) sum += input[i + blockDim.x];
    
    sdata[tid] = sum;
    __syncthreads();
    
    // Reduce in shared memory
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Reduce final warp
    if (tid < 32) {
        sdata[tid] = warp_reduce_sum(sdata[tid]);
    }
    
    // Write result for this block to global memory
    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Complete reduction with multiple launches
template<typename T>
void reduce_complete(T* d_in, T* d_out, int n) {
    const int threads = 512;
    const int blocks = (n + threads - 1) / threads;
    const int shared_mem_size = threads * sizeof(T);
    
    reduction_kernel<<<blocks, threads, shared_mem_size>>>(d_in, d_out, n);
    
    // If we have more than one block, recursively reduce
    if (blocks > 1) {
        reduce_complete(d_out, d_out, blocks);
    }
}
```

### Solution 3: Memory Access Pattern Analysis
```cuda
// Problematic kernel with poor memory access
__global__ void poor_access_pattern(float* matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    // This creates poor coalescing for column-wise access
    if (row < cols && col < rows) {  // Note: swapped row/col
        matrix[col * cols + row] = matrix[col * cols + row] * 2.0f;
    }
}

// Optimized version with proper coalescing
__global__ void optimized_access_pattern(float* matrix, int rows, int cols) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Proper indexing for row-major access
    if (row < rows && col < cols) {
        matrix[row * cols + col] = matrix[row * cols + col] * 2.0f;
    }
}

// Alternative: transposed access pattern if needed
__global__ void column_optimized_pattern(float* matrix, int rows, int cols) {
    // Process in tiles to optimize for column access
    const int TILE_SIZE = 16;
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid bank conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load in row-major order
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] = 
                matrix[(y + j) * cols + x];
        }
    }
    __syncthreads();
    
    // Process in column-major order from shared memory
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (x < cols && y + j < rows) {
            tile[threadIdx.y + j][threadIdx.x] *= 2.0f;
        }
    }
    __syncthreads();
    
    // Write back in row-major order
    for (int j = 0; j < TILE_SIZE; j += blockDim.y) {
        if (x < cols && y + j < rows) {
            matrix[(y + j) * cols + x] = 
                tile[threadIdx.y + j][threadIdx.x];
        }
    }
}
```

## Advanced Topic: CUTLASS Connection

Understanding how these CUDA fundamentals connect to CUTLASS:

```cuda
// Simplified example showing CUTLASS concepts
namespace cutlass_concepts {

// Threadblock-level operations
__global__ void threadblock_gemm(
    const float* A, const float* B, float* C,
    int M, int N, int K) {
    
    // Each block handles a tile of the computation
    int block_m = blockIdx.y * 128;  // Example tile size
    int block_n = blockIdx.x * 128;
    
    // Each thread in block handles a portion of the tile
    int thread_m = block_m + threadIdx.y * 8;  // Example thread tile
    int thread_n = block_n + threadIdx.x * 8;
    
    // Accumulate partial products
    float accumulator[8][8] = {0};  // Local accumulation
    
    // Iterate through K dimension in tiles
    for (int k = 0; k < K; k += 32) {  // Example K tile size
        
        // Load fragments from global to shared memory
        // Process fragments at warp/warp-group level
        // Accumulate results
    }
    
    // Write results back to global memory
}

// CUTLASS-style memory layout
struct ColumnMajor {
    __host__ __device__
    int operator()(int row, int col, int ld) const {
        return col * ld + row;  // Column-major indexing
    }
};

struct RowMajor {
    __host__ __device__
    int operator()(int row, int col, int ld) const {
        return row * ld + col;  // Row-major indexing
    }
};

} // namespace cutlass_concepts
```

## Quiz Questions

1. What is the difference between a CUDA grid, block, and thread?

2. Explain the concept of warps and why they're important for GPU performance.

3. What is memory coalescing and why is it important?

4. How do shared memory banks work and what are bank conflicts?

5. What factors affect occupancy in CUDA kernels?

## Summary
Module 5 covered essential CUDA and GPU programming fundamentals including GPU architecture, the CUDA programming model, thread organization, memory hierarchies, coalesced memory access, and performance optimization. These concepts are crucial for understanding how CUTLASS leverages GPU architecture for high-performance linear algebra operations.