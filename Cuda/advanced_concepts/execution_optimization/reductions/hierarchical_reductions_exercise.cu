/*
 * Hierarchical Reductions Exercise
 *
 * This exercise demonstrates how to implement multi-block hierarchical reductions
 * that can handle large arrays by combining results from multiple blocks.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Kernel 1: Block-level Reduction (First Phase)
__global__ void blockReduction(float* input, float* temp, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        temp[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Grid-level Reduction (Second Phase)
__global__ void gridReduction(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Kernel 3: Complete Hierarchical Reduction (Multi-phase)
__global__ void hierarchicalReduction(float* input, float* temp, float* output, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // First phase: block-level reduction
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to temporary array
    if (tid == 0) {
        temp[blockIdx.x] = sdata[0];
    }
    
    // Second phase: grid-level reduction (only for the first block after sync)
    __syncthreads();
    
    // If this is the first block and we have a single block for the final reduction
    if (blockIdx.x == 0 && gridDim.x == 1) {
        // This is just the first phase again, in a real hierarchical reduction
        // the second phase would be launched separately
    }
}

// Kernel 4: Student Exercise - Implement multi-phase hierarchical reduction
__global__ void studentHierarchicalReduction(float* input, float* temp, float* output, int n) {
    // TODO: Implement a complete multi-phase hierarchical reduction
    // HINT: First reduce within each block, then reduce the block results
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Phase 1: Block-level reduction
    sdata[tid] = (i < n) ? input[i] : 0.0f;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to temporary array
    if (tid == 0) {
        temp[blockIdx.x] = sdata[0];
    }
    
    // Phase 2: Grid-level reduction - this would normally be a separate kernel launch
    // For this exercise, implement a way to handle the multi-phase nature
    // FIX: Implement logic to detect when all block-level reductions are complete
    // and then perform the final reduction of the partial results
    
    // In a real implementation, you would:
    // 1. Launch this kernel to get partial results in 'temp'
    // 2. Check if temp has more than one element
    // 3. If yes, launch this kernel again with temp as input
    // 4. Repeat until you have a single result
}

// Kernel 5: Student Exercise - Implement hierarchical reduction with arbitrary size
__global__ void studentArbitrarySizeReduction(float* input, float* temp, float* output, int n) {
    // TODO: Implement hierarchical reduction that can handle any size input
    // HINT: Handle cases where n is not a power of 2 or doesn't align with block size
    
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIX: Handle arbitrary size input
    // The standard approach might leave gaps or have unbalanced workloads
    sdata[tid] = 0.0f;  // Initialize to 0
    
    // FIX: Load multiple elements per thread if n > gridDim.x * blockDim.x
    // This is called "grid-stride loop" approach
    float sum = 0.0f;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        sum += input[idx];
    }
    sdata[tid] = sum;
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result for this block to temporary array
    if (tid == 0) {
        temp[blockIdx.x] = sdata[0];
    }
}

// Kernel 6: Student Exercise - Implement hierarchical reduction with different operations
__global__ void studentMultiOpHierarchical(float* input, float* temp_sum, float* temp_max, 
                                       float* output_sum, float* output_max, int n) {
    // TODO: Implement hierarchical reduction that computes multiple operations (sum and max)
    // HINT: Use separate temporary arrays for each operation
    
    extern __shared__ float sdata_sum[];
    extern __shared__ float sdata_max[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // FIX: Handle the challenge that extern __shared__ can't be declared twice
    // Solution: Use a single shared memory array and partition it
    float* sdata = sdata_sum;  // Use the same memory region
    float* sdata_max_region = &sdata[blockDim.x];  // Use second half for max values
    
    // Initialize
    float my_sum = (i < n) ? input[i] : 0.0f;
    float my_max = (i < n) ? input[i] : -FLT_MAX;
    
    // Store in shared memory
    sdata[tid] = my_sum;
    sdata_max_region[tid] = my_max;
    __syncthreads();
    
    // Perform sum reduction
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Perform max reduction
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            sdata_max_region[tid] = fmaxf(sdata_max_region[tid], sdata_max_region[tid + s]);
        }
        __syncthreads();
    }
    
    // Write results for this block to temporary arrays
    if (tid == 0) {
        temp_sum[blockIdx.x] = sdata[0];
        temp_max[blockIdx.x] = sdata_max_region[0];
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 100) * 0.1f;
    }
}

int main() {
    printf("=== Hierarchical Reductions Exercise ===\n");
    printf("Learn to implement multi-block hierarchical reduction algorithms.\n\n");

    // Setup parameters
    const int N = 1024 * 16;  // Multiple of block size
    const int blockSize = 256;
    const int gridSize = (N + blockSize - 1) / blockSize;
    size_t bytes = N * sizeof(float);
    size_t temp_bytes = gridSize * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_temp, *h_output;
    h_input = (float*)malloc(bytes);
    h_temp = (float*)malloc(temp_bytes);
    h_output = (float*)malloc(sizeof(float));  // Final result
    
    // Initialize data
    initArray(h_input, N, 1.0f);
    
    // Allocate device memory
    float *d_input, *d_temp, *d_output;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_temp, temp_bytes);
    cudaMalloc(&d_output, sizeof(float));
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Define shared memory size
    size_t shared_mem_size = blockSize * sizeof(float);
    
    // Run block-level reduction kernel (first phase)
    printf("Running block-level reduction (phase 1)...\n");
    blockReduction<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_temp, N);
    cudaDeviceSynchronize();
    
    // Calculate number of blocks needed for second phase
    int second_phase_blocks = (gridSize + blockSize - 1) / blockSize;
    size_t second_temp_bytes = second_phase_blocks * sizeof(float);
    float *d_second_temp;
    cudaMalloc(&d_second_temp, second_temp_bytes);
    
    // Run grid-level reduction kernel (second phase)
    printf("Running grid-level reduction (phase 2)...\n");
    gridReduction<<<second_phase_blocks, blockSize, shared_mem_size>>>(d_temp, d_second_temp, gridSize);
    cudaDeviceSynchronize();
    
    // If we have more than one result after second phase, continue reduction
    if (second_phase_blocks > 1) {
        int third_phase_blocks = (second_phase_blocks + blockSize - 1) / blockSize;
        gridReduction<<<third_phase_blocks, blockSize, shared_mem_size>>>(d_second_temp, d_output, second_phase_blocks);
        cudaDeviceSynchronize();
    } else {
        cudaMemcpy(d_output, d_second_temp, sizeof(float), cudaMemcpyDeviceToDevice);
    }
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student hierarchical reduction exercises (complete the code first!)...\n");
    
    // Hierarchical reduction exercise
    studentHierarchicalReduction<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_temp, d_output, N);
    cudaDeviceSynchronize();
    
    // Arbitrary size reduction exercise
    studentArbitrarySizeReduction<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_temp, d_output, N);
    cudaDeviceSynchronize();
    
    // Multi-operation hierarchical reduction exercise
    float *d_temp_max;
    cudaMalloc(&d_temp_max, temp_bytes);
    studentMultiOpHierarchical<<<gridSize, blockSize, 2 * shared_mem_size>>>(d_input, d_temp, d_temp_max, d_output, d_temp_max, N);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the hierarchical reduction implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy final result back to host
    cudaMemcpy(h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);
    
    // Calculate expected result for verification
    float expected_sum = 0.0f;
    for (int i = 0; i < N; i++) {
        expected_sum += h_input[i];
    }
    
    // Print results
    printf("\nResults:\n");
    printf("Expected sum: %.2f\n", expected_sum);
    printf("Computed sum: %.2f\n", h_output[0]);
    printf("Difference: %.2f\n", fabsf(expected_sum - h_output[0]));
    
    // Cleanup
    free(h_input); free(h_temp); free(h_output);
    cudaFree(d_input); cudaFree(d_temp); cudaFree(d_output); 
    cudaFree(d_second_temp); cudaFree(d_temp_max);
    
    printf("\nExercise completed! Notice how hierarchical reductions handle large arrays.\n");
    
    return 0;
}