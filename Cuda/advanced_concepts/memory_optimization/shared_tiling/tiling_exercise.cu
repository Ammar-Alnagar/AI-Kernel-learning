/*
 * Shared Memory Tiling Exercise
 *
 * This exercise demonstrates how to use shared memory tiling to improve memory
 * access patterns and performance for matrix operations.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

#define TILE_SIZE 16

// Kernel 1: Naive Matrix Multiplication (Without Tiling)
__global__ void naiveMatMul(float* A, float* B, float* C, int width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < width && col < width) {
        float sum = 0.0f;
        for (int k = 0; k < width; k++) {
            sum += A[row * width + k] * B[k * width + col];
        }
        C[row * width + col] = sum;
    }
}

// Kernel 2: Tiled Matrix Multiplication (With Shared Memory)
__global__ void tiledMatMul(float* A, float* B, float* C, int width) {
    // Shared memory tiles
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    
    // Thread's final result
    float Csub = 0.0f;
    
    // Loop over tiles
    for (int m = 0; m < (width + TILE_SIZE - 1) / TILE_SIZE; m++) {
        // Collaborative loading of tiles into shared memory
        if (bx * TILE_SIZE + tx < width && by * TILE_SIZE + ty < width) {
            As[ty][tx] = A[(by * TILE_SIZE + ty) * width + (bx * TILE_SIZE + tx)];
        } else {
            As[ty][tx] = 0.0f;
        }
        
        if (bx * TILE_SIZE + tx < width && m * TILE_SIZE + ty < width) {
            Bs[ty][tx] = B[(m * TILE_SIZE + ty) * width + (bx * TILE_SIZE + tx)];
        } else {
            Bs[ty][tx] = 0.0f;
        }
        
        __syncthreads();
        
        // Compute partial result for this tile
        for (int k = 0; k < TILE_SIZE; k++) {
            Csub += As[ty][k] * Bs[k][tx];
        }
        
        __syncthreads();
    }
    
    // Write result to global memory
    int row = by * TILE_SIZE + ty;
    int col = bx * TILE_SIZE + tx;
    if (row < width && col < width) {
        C[row * width + col] = Csub;
    }
}

// Kernel 3: Student Exercise - Implement tiling for transpose operation
__global__ void studentTiledTranspose(float* input, float* output, int width) {
    // TODO: Declare shared memory tiles for tiling
    // Hint: You'll need two tiles - one for input and one for transposed output
    /* YOUR SHARED MEMORY DECLARATIONS HERE */;
    
    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    // TODO: Load tile from input to shared memory
    // Remember to handle boundary conditions
    if (/* YOUR CONDITION */) {
        tile_in[threadIdx.y][threadIdx.x] = input[y * width + x];
    } else {
        tile_in[threadIdx.y][threadIdx.x] = 0.0f;  // Padding
    }
    __syncthreads();
    
    // TODO: Perform transposed write from shared memory to output
    // Remember to swap x and y coordinates for transpose
    x = blockIdx.y * TILE_SIZE + threadIdx.x;  // Swapped blocks for output
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (/* YOUR CONDITION */) {
        // FIX: Implement the transposed write to avoid bank conflicts
        // Consider adding padding to shared memory to avoid bank conflicts
        output[y * width + x] = /* YOUR TRANSPOSED ACCESS */;
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int width, float start_val = 1.0f) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = start_val + (i % 1000) * 0.01f;
    }
}

// Utility function to print matrix
void printMatrix(float* mat, int width, int rows = 4, int cols = 4) {
    printf("First %d×%d elements:\n", rows, cols);
    for (int i = 0; i < rows && i < width; i++) {
        for (int j = 0; j < cols && j < width; j++) {
            printf("%6.1f ", mat[i * width + j]);
        }
        printf("\n");
    }
}

int main() {
    printf("=== Shared Memory Tiling Exercise ===\n");
    printf("Compare naive vs tiled matrix multiplication and implement tiled transpose.\n\n");

    // Setup parameters
    const int WIDTH = 512;  // Must be multiple of TILE_SIZE for simplicity
    const int SIZE = WIDTH * WIDTH;
    size_t bytes = SIZE * sizeof(float);
    
    // Allocate host memory
    float *h_A, *h_B, *h_C_naive, *h_C_tiled, *h_input_trans, *h_output_trans;
    h_A = (float*)malloc(bytes);
    h_B = (float*)malloc(bytes);
    h_C_naive = (float*)malloc(bytes);
    h_C_tiled = (float*)malloc(bytes);
    h_input_trans = (float*)malloc(bytes);
    h_output_trans = (float*)malloc(bytes);
    
    // Initialize matrices
    initMatrix(h_A, WIDTH, 1.0f);
    initMatrix(h_B, WIDTH, 2.0f);
    initMatrix(h_input_trans, WIDTH, 3.0f);
    
    // Allocate device memory
    float *d_A, *d_B, *d_C_naive, *d_C_tiled, *d_input_trans, *d_output_trans;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C_naive, bytes);
    cudaMalloc(&d_C_tiled, bytes);
    cudaMalloc(&d_input_trans, bytes);
    cudaMalloc(&d_output_trans, bytes);
    
    // Copy input to device
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input_trans, h_input_trans, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((WIDTH + TILE_SIZE - 1) / TILE_SIZE, 
                  (WIDTH + TILE_SIZE - 1) / TILE_SIZE);
    
    // Run naive matrix multiplication
    printf("Running naive matrix multiplication...\n");
    naiveMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C_naive, WIDTH);
    cudaDeviceSynchronize();
    
    // Run tiled matrix multiplication
    printf("Running tiled matrix multiplication...\n");
    tiledMatMul<<<gridSize, blockSize>>>(d_A, d_B, d_C_tiled, WIDTH);
    cudaDeviceSynchronize();
    
    // Run student exercise (will fail to compile until completed)
    printf("Running student tiling exercise (complete the code first!)...\n");
    studentTiledTranspose<<<gridSize, blockSize>>>(d_input_trans, d_output_trans, WIDTH);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the shared memory declarations and tiling logic in studentTiledTranspose!\n");
    } else {
        printf("Student exercise kernel executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_C_naive, d_C_naive, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_tiled, d_C_tiled, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_trans, d_output_trans, bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results from matrix multiplication (first 4 elements):\n");
    printf("Naive: %.1f %.1f %.1f %.1f\n", 
           h_C_naive[0], h_C_naive[1], h_C_naive[2], h_C_naive[3]);
    printf("Tiled: %.1f %.1f %.1f %.1f\n", 
           h_C_tiled[0], h_C_tiled[1], h_C_tiled[2], h_C_tiled[3]);
    
    printf("\nSample results from transpose (first 4 elements of first row):\n");
    printf("Input:   %.1f %.1f %.1f %.1f\n", 
           h_input_trans[0], h_input_trans[1], h_input_trans[2], h_input_trans[3]);
    printf("Output:  %.1f %.1f %.1f %.1f\n", 
           h_output_trans[0], h_output_trans[WIDTH], h_output_trans[WIDTH*2], h_output_trans[WIDTH*3]);
    
    // Cleanup
    free(h_A); free(h_B); free(h_C_naive); free(h_C_tiled); 
    free(h_input_trans); free(h_output_trans);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C_naive); cudaFree(d_C_tiled);
    cudaFree(d_input_trans); cudaFree(d_output_trans);
    
    printf("\nExercise completed! Notice how tiling improves memory access patterns.\n");
    
    return 0;
}