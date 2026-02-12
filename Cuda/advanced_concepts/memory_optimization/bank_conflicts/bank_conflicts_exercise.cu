/*
 * Shared Memory Bank Conflicts Exercise
 *
 * This exercise demonstrates how to identify and fix shared memory bank conflicts
 * in CUDA kernels.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define WARP_SIZE 32
#define TILE_SIZE 32

// Kernel 1: With Bank Conflicts (INEFFICIENT)
__global__ void bankConflictTranspose(float* input, float* output, int width) {
    // Shared memory tile - causes bank conflicts during transposed access
    __shared__ float tile[TILE_SIZE][TILE_SIZE];
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory (coalesced read)
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Transpose write - CAUSES BANK CONFLICTS
    // When threads access tile[threadIdx.x][threadIdx.y], multiple threads 
    // may access the same memory bank simultaneously
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < width && y < width) {
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Kernel 2: Without Bank Conflicts (EFFICIENT)
__global__ void noBankConflictTranspose(float* input, float* output, int width) {
    // Shared memory tile with padding to avoid bank conflicts
    __shared__ float tile[TILE_SIZE][TILE_SIZE + 1];  // +1 to avoid conflicts
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // Load data into shared memory (coalesced read)
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    // Transpose write - NO BANK CONFLICTS due to padding
    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < width && y < width) {
        output[y * width + x] = tile[threadIdx.x][threadIdx.y];
    }
}

// Kernel 3: Student Exercise - Fix bank conflicts in reduction
__global__ void studentBankConflictReduction(float* input, float* output, int n) {
    // TODO: Fix bank conflicts in this reduction implementation
    // Current implementation uses a 2D shared memory array that may cause conflicts
    
    // Current (conflicted) shared memory declaration:
    // __shared__ float sdata[TILE_SIZE][TILE_SIZE];  // This causes conflicts
    
    // FIX: Declare shared memory to avoid bank conflicts
    // Hint: Consider how threads access the shared memory and add padding if needed
    /* YOUR FIXED SHARED MEMORY DECLARATION */;
    
    int tid = threadIdx.x;
    int bid = blockIdx.x;
    int i = bid * blockDim.x + threadIdx.x;
    
    // Load input into shared memory
    if (i < n) {
        // FIX: Store in shared memory considering the new declaration
        /* YOUR FIXED STORAGE CODE */;
    } else {
        // FIX: Handle padding case
        /* YOUR PADDING CODE */;
    }
    __syncthreads();
    
    // Perform reduction in shared memory
    for (int s = 1; s < blockDim.x; s *= 2) {
        if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
            // FIX: Access shared memory considering the new declaration
            /* YOUR FIXED REDUCTION ACCESS */;
        }
        __syncthreads();
    }
    
    // Write result for this block to global memory
    if (tid == 0) {
        output[bid] = /* YOUR RESULT ACCESS */;
    }
}

// Alternative student exercise: Fix bank conflicts in gather operation
__global__ void studentGatherOperation(int* indices, float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        int idx = indices[tid];
        
        // Shared memory for caching - potential bank conflicts
        __shared__ float cache[WARP_SIZE * 2];  // Potential conflicts if consecutive threads access consecutive banks
        
        // TODO: Fix the indexing to avoid bank conflicts
        // Current: cache[threadIdx.x] - causes conflicts when consecutive threads access consecutive locations
        // FIX: Modify the indexing to avoid conflicts
        cache[/* YOUR FIXED INDEX */] = input[idx];
        __syncthreads();
        
        // Store result
        output[tid] = cache[/* YOUR FIXED INDEX */];
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 100) * 0.1f;
    }
}

// Utility function to initialize indices
void initIndices(int* indices, int n) {
    for (int i = 0; i < n; i++) {
        indices[i] = i;  // Sequential for predictable access
    }
}

int main() {
    printf("=== Shared Memory Bank Conflicts Exercise ===\n");
    printf("Learn to identify and fix shared memory bank conflicts.\n\n");

    // Setup parameters for transpose
    const int WIDTH = 128;
    const int SIZE = WIDTH * WIDTH;
    size_t bytes = SIZE * sizeof(float);
    
    // Allocate host memory for transpose
    float *h_input, *h_output_conflict, *h_output_no_conflict;
    h_input = (float*)malloc(bytes);
    h_output_conflict = (float*)malloc(bytes);
    h_output_no_conflict = (float*)malloc(bytes);
    
    // Initialize input matrix
    initArray(h_input, SIZE, 1.0f);
    
    // Allocate device memory for transpose
    float *d_input, *d_output_conflict, *d_output_no_conflict;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output_conflict, bytes);
    cudaMalloc(&d_output_no_conflict, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions for transpose
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((WIDTH + TILE_SIZE - 1) / TILE_SIZE, 
                  (WIDTH + TILE_SIZE - 1) / TILE_SIZE);
    
    // Run transpose with bank conflicts
    printf("Running transpose with bank conflicts...\n");
    bankConflictTranspose<<<gridSize, blockSize>>>(d_input, d_output_conflict, WIDTH);
    cudaDeviceSynchronize();
    
    // Run transpose without bank conflicts
    printf("Running transpose without bank conflicts...\n");
    noBankConflictTranspose<<<gridSize, blockSize>>>(d_input, d_output_no_conflict, WIDTH);
    cudaDeviceSynchronize();
    
    // Setup parameters for reduction exercise
    const int REDUCE_N = 1024;
    size_t reduce_bytes = REDUCE_N * sizeof(float);
    size_t reduce_out_bytes = ((REDUCE_N + 255) / 256) * sizeof(float); // Assuming 256 threads per block
    
    // Allocate memory for reduction
    float *h_input_red, *h_output_red;
    int *h_indices;
    h_input_red = (float*)malloc(reduce_bytes);
    h_output_red = (float*)malloc(reduce_out_bytes);
    h_indices = (int*)malloc(REDUCE_N * sizeof(int));
    
    // Initialize data for reduction
    initArray(h_input_red, REDUCE_N, 1.0f);
    initIndices(h_indices, REDUCE_N);
    
    // Allocate device memory for reduction
    float *d_input_red, *d_output_red;
    int *d_indices;
    cudaMalloc(&d_input_red, reduce_bytes);
    cudaMalloc(&d_output_red, reduce_out_bytes);
    cudaMalloc(&d_indices, REDUCE_N * sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_input_red, h_input_red, reduce_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_indices, h_indices, REDUCE_N * sizeof(int), cudaMemcpyHostToDevice);
    
    // Run student exercise kernels (will fail to compile until completed)
    printf("Running student bank conflict exercises (complete the code first!)...\n");
    
    // Reduction exercise
    int reduce_block_size = 256;
    int reduce_grid_size = (REDUCE_N + reduce_block_size - 1) / reduce_block_size;
    studentBankConflictReduction<<<reduce_grid_size, reduce_block_size>>>(d_input_red, d_output_red, REDUCE_N);
    cudaDeviceSynchronize();
    
    // Gather operation exercise
    studentGatherOperation<<<reduce_grid_size, reduce_block_size>>>(d_indices, d_input_red, d_output_red, REDUCE_N);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the bank conflict fixes in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_conflict, d_output_conflict, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_no_conflict, d_output_no_conflict, bytes, cudaMemcpyDeviceToHost);
    
    // Cleanup
    free(h_input); free(h_output_conflict); free(h_output_no_conflict);
    free(h_input_red); free(h_output_red); free(h_indices);
    cudaFree(d_input); cudaFree(d_output_conflict); cudaFree(d_output_no_conflict);
    cudaFree(d_input_red); cudaFree(d_output_red); cudaFree(d_indices);
    
    printf("\nExercise completed! Notice how padding can eliminate bank conflicts.\n");
    
    return 0;
}