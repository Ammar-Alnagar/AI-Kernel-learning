/*
 * Softmax Exercise
 *
 * This exercise demonstrates how to implement efficient softmax operations,
 * including naive, online, and tiled approaches.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define LOG_SUM_EXP_THRESHOLD 20.0f

// Kernel 1: Naive Softmax (INEFFICIENT and numerically unstable)
__global__ void naiveSoftmax(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Find max for numerical stability
        float max_val = input[row * cols];
        for (int j = 1; j < cols; j++) {
            if (input[row * cols + j] > max_val) {
                max_val = input[row * cols + j];
            }
        }
        
        // Compute exponentials and sum
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            float exp_val = expf(input[row * cols + j] - max_val);
            sum += exp_val;
        }
        
        // Normalize
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - max_val) / sum;
        }
    }
}

// Kernel 2: Online Softmax (NUMERICALLY STABLE)
__global__ void onlineSoftmax(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Two-pass approach for numerical stability
        // Pass 1: Find max
        float max_val = input[row * cols];
        for (int j = 1; j < cols; j++) {
            if (input[row * cols + j] > max_val) {
                max_val = input[row * cols + j];
            }
        }
        
        // Pass 2: Compute sum of exponentials and normalize
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(input[row * cols + j] - max_val);
        }
        
        // Normalize
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - max_val) / sum;
        }
    }
}

// Kernel 3: Student Exercise - Implement block-level softmax with shared memory
__global__ void studentBlockSoftmax(float* input, float* output, int rows, int cols) {
    // TODO: Implement block-level softmax using shared memory for reductions
    // HINT: Use shared memory to find max and sum within each block
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    
    // FIX: Use shared memory to coordinate within the block
    // Since each row is independent, we can process one row per block
    // But we'll need to coordinate within the block to compute max and sum
    
    if (row < rows) {
        // FIX: Use shared memory for reduction operations
        extern __shared__ float sdata[];
        
        float max_val = -INFINITY;
        float sum = 0.0f;
        
        // FIX: Implement block-level reduction to find max
        // Each thread handles multiple elements if cols > blockDim.x
        int elements_per_thread = (cols + blockDim.x - 1) / blockDim.x;
        int start_col = tid * elements_per_thread;
        int end_col = min(start_col + elements_per_thread, cols);
        
        // Find local max for this thread
        float local_max = -INFINITY;
        for (int j = start_col; j < end_col; j++) {
            if (input[row * cols + j] > local_max) {
                local_max = input[row * cols + j];
            }
        }
        
        // FIX: Use shared memory to find global max across threads in block
        sdata[tid] = local_max;
        __syncthreads();
        
        // Perform reduction to find max
        for (int s = 1; s < blockDim.x; s *= 2) {
            if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
                sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
            }
            __syncthreads();
        }
        
        max_val = sdata[0];
        __syncthreads();
        
        // FIX: Now compute sum of exponentials using the found max
        float local_sum = 0.0f;
        for (int j = start_col; j < end_col; j++) {
            local_sum += expf(input[row * cols + j] - max_val);
        }
        
        // FIX: Use shared memory to find global sum across threads in block
        sdata[tid] = local_sum;
        __syncthreads();
        
        // Perform reduction to find sum
        for (int s = 1; s < blockDim.x; s *= 2) {
            if ((tid % (2*s)) == 0 && (tid + s) < blockDim.x) {
                sdata[tid] += sdata[tid + s];
            }
            __syncthreads();
        }
        
        sum = sdata[0];
        __syncthreads();
        
        // Apply softmax
        for (int j = start_col; j < end_col; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - max_val) / sum;
        }
    }
}

// Kernel 4: Student Exercise - Implement tiled softmax for very wide matrices
__global__ void studentTiledSoftmax(float* input, float* output, int rows, int cols) {
    // TODO: Implement tiled softmax for cases where cols is very large
    // HINT: Process the row in tiles to handle cases where cols > shared memory capacity
    
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // FIX: Implement tiled approach for very wide rows
        // Process the row in chunks to handle large numbers of columns
        
        // For this exercise, implement a simplified version assuming we can fit
        // the entire row in shared memory (though in practice, you'd need tiling)
        
        // FIX: Use a tile size that fits in shared memory
        const int TILE_SIZE = 256;  // Adjust based on shared memory limits
        
        // FIX: Implement the algorithm using tiles
        // You'll need to process the row in chunks and combine results
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int rows, int cols, float start_val = 1.0f) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            // Initialize with values that will make softmax meaningful
            mat[i * cols + j] = start_val + (i + j) * 0.1f - 5.0f;  // Center around 0
        }
    }
}

// Utility function to print matrix
void printMatrix(float* mat, int rows, int cols, int print_rows = 2, int print_cols = 5) {
    printf("First %d×%d elements:\n", print_rows, print_cols);
    for (int i = 0; i < print_rows && i < rows; i++) {
        for (int j = 0; j < print_cols && j < cols; j++) {
            printf("%6.3f ", mat[i * cols + j]);
        }
        printf("\n");
    }
}

// Function to validate softmax output (each row should sum to 1.0)
bool validateSoftmax(float* output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        float row_sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            row_sum += output[i * cols + j];
        }
        if (fabsf(row_sum - 1.0f) > 1e-3) {
            printf("Row %d sum is %f (should be 1.0)\n", i, row_sum);
            return false;
        }
    }
    return true;
}

int main() {
    printf("=== Softmax Exercise ===\n");
    printf("Learn to implement efficient and numerically stable softmax operations.\n\n");

    // Setup parameters
    const int ROWS = 256;
    const int COLS = 128;
    size_t row_bytes = COLS * sizeof(float);
    size_t total_bytes = ROWS * COLS * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output_naive, *h_output_online, *h_output_block, *h_output_tiled;
    h_input = (float*)malloc(total_bytes);
    h_output_naive = (float*)malloc(total_bytes);
    h_output_online = (float*)malloc(total_bytes);
    h_output_block = (float*)malloc(total_bytes);
    h_output_tiled = (float*)malloc(total_bytes);
    
    // Initialize matrix
    initMatrix(h_input, ROWS, COLS, 0.0f);
    
    // Allocate device memory
    float *d_input, *d_output_naive, *d_output_online, *d_output_block, *d_output_tiled;
    cudaMalloc(&d_input, total_bytes);
    cudaMalloc(&d_output_naive, total_bytes);
    cudaMalloc(&d_output_online, total_bytes);
    cudaMalloc(&d_output_block, total_bytes);
    cudaMalloc(&d_output_tiled, total_bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, total_bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (ROWS + blockSize - 1) / blockSize;
    size_t shared_mem_size = blockSize * sizeof(float);  // For block-level reductions
    
    // Run naive softmax kernel
    printf("Running naive softmax kernel...\n");
    naiveSoftmax<<<gridSize, blockSize>>>(d_input, d_output_naive, ROWS, COLS);
    cudaDeviceSynchronize();
    
    // Run online softmax kernel
    printf("Running online softmax kernel...\n");
    onlineSoftmax<<<gridSize, blockSize>>>(d_input, d_output_online, ROWS, COLS);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student softmax exercises (complete the code first!)...\n");
    
    // Block-level softmax exercise
    studentBlockSoftmax<<<gridSize, blockSize, shared_mem_size>>>(d_input, d_output_block, ROWS, COLS);
    cudaDeviceSynchronize();
    
    // Tiled softmax exercise
    studentTiledSoftmax<<<gridSize, blockSize>>>(d_input, d_output_tiled, ROWS, COLS);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the softmax implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_naive, d_output_naive, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_online, d_output_online, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_block, d_output_block, total_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_tiled, d_output_tiled, total_bytes, cudaMemcpyDeviceToHost);
    
    // Validate results
    printf("\nValidation results:\n");
    printf("Naive softmax valid: %s\n", validateSoftmax(h_output_naive, ROWS, COLS) ? "YES" : "NO");
    printf("Online softmax valid: %s\n", validateSoftmax(h_output_online, ROWS, COLS) ? "YES" : "NO");
    printf("Block softmax valid: %s\n", validateSoftmax(h_output_block, ROWS, COLS) ? "YES" : "NO");
    
    // Print sample results
    printf("\nSample results (first row):\n");
    printf("Input:    ");
    for (int j = 0; j < 5; j++) {
        printf("%6.2f ", h_input[j]);
    }
    printf("\n");
    printf("Naive:    ");
    for (int j = 0; j < 5; j++) {
        printf("%6.3f ", h_output_naive[j]);
    }
    printf("\n");
    printf("Online:   ");
    for (int j = 0; j < 5; j++) {
        printf("%6.3f ", h_output_online[j]);
    }
    printf("\n");
    printf("Block:    ");
    for (int j = 0; j < 5; j++) {
        printf("%6.3f ", h_output_block[j]);
    }
    printf("\n");
    
    // Cleanup
    free(h_input); free(h_output_naive); free(h_output_online); 
    free(h_output_block); free(h_output_tiled);
    cudaFree(d_input); cudaFree(d_output_naive); cudaFree(d_output_online);
    cudaFree(d_output_block); cudaFree(d_output_tiled);
    
    printf("\nExercise completed! Notice how numerical stability matters in softmax.\n");
    
    return 0;
}