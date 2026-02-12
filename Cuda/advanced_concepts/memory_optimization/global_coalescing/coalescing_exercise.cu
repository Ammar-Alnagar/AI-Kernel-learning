/*
 * Global Memory Coalescing Exercise
 *
 * This exercise demonstrates the importance of coalesced memory access patterns
 * and how they affect GPU performance.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

// Kernel 1: Uncoalesced Access Pattern (INEFFICIENT)
__global__ void uncoalescedAccess(float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        // UNCOALESCED: Strided access pattern - threads access memory with stride
        // This causes poor memory bandwidth utilization
        int idx = col * height + row;  // Column-major access in row-major data
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel 2: Coalesced Access Pattern (EFFICIENT)
__global__ void coalescedAccess(float* input, float* output, int width, int height) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < height && col < width) {
        // COALESCED: Contiguous access pattern - consecutive threads access consecutive memory
        // This maximizes memory bandwidth utilization
        int idx = row * width + col;  // Row-major access in row-major data
        output[idx] = input[idx] * 2.0f;
    }
}

// Kernel 3: Student Exercise - Fix the inefficient access pattern
__global__ void studentExercise(float* input, float* output, int width, int height, int stride) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    // TODO: Fix this inefficient access pattern to be coalesced
    // Current implementation: threads access memory with stride (inefficient)
    // int idx = tid * stride;  // This creates strided access
    
    // FIX: Implement coalesced access where consecutive threads access consecutive memory
    int idx = /* YOUR COALESCED ACCESS HERE */;
    
    if (idx < width * height) {
        output[idx] = input[idx] * 3.0f + 1.0f;
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int width, int height) {
    int size = width * height;
    for (int i = 0; i < size; i++) {
        mat[i] = (float)(i % 1000) / 100.0f;
    }
}

// Utility function to measure time
double getTime() {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int main() {
    printf("=== Global Memory Coalescing Exercise ===\n");
    printf("Compare coalesced vs uncoalesced memory access patterns.\n\n");

    // Setup parameters
    const int WIDTH = 1024;
    const int HEIGHT = 1024;
    const int SIZE = WIDTH * HEIGHT;
    const int STRIDE = 32;  // For student exercise
    size_t bytes = SIZE * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_output_uncoal, *h_output_coal, *h_output_student;
    h_input = (float*)malloc(bytes);
    h_output_uncoal = (float*)malloc(bytes);
    h_output_coal = (float*)malloc(bytes);
    h_output_student = (float*)malloc(bytes);
    
    // Initialize input matrix
    initMatrix(h_input, WIDTH, HEIGHT);
    
    // Allocate device memory
    float *d_input, *d_output_uncoal, *d_output_coal, *d_output_student;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output_uncoal, bytes);
    cudaMalloc(&d_output_coal, bytes);
    cudaMalloc(&d_output_student, bytes);
    
    // Copy input to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);
    dim3 gridSize((WIDTH + blockSize.x - 1) / blockSize.x, 
                  (HEIGHT + blockSize.y - 1) / blockSize.y);
    
    // Timing variables
    double start, end;
    
    // Run uncoalesced kernel
    printf("Running uncoalesced access kernel...\n");
    start = getTime();
    uncoalescedAccess<<<gridSize, blockSize>>>(d_input, d_output_uncoal, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    end = getTime();
    printf("Uncoalesced kernel time: %.4f seconds\n", end - start);
    
    // Run coalesced kernel
    printf("Running coalesced access kernel...\n");
    start = getTime();
    coalescedAccess<<<gridSize, blockSize>>>(d_input, d_output_coal, WIDTH, HEIGHT);
    cudaDeviceSynchronize();
    end = getTime();
    printf("Coalesced kernel time: %.4f seconds\n", end - start);
    
    // Run student exercise kernel (will fail to compile until completed)
    printf("Running student exercise kernel (complete the code first!)...\n");
    int linearSize = WIDTH * HEIGHT;
    int linearBlockSize = 256;
    int linearGridSize = (linearSize + linearBlockSize - 1) / linearBlockSize;
    
    studentExercise<<<linearGridSize, linearBlockSize>>>(d_input, d_output_student, WIDTH, HEIGHT, STRIDE);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the coalesced access pattern in the studentExercise kernel!\n");
    } else {
        printf("Student exercise kernel executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_uncoal, d_output_uncoal, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_coal, d_output_coal, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_student, d_output_student, bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Input:     %.2f %.2f %.2f %.2f %.2f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    printf("Coalesced: %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_coal[0], h_output_coal[1], h_output_coal[2], h_output_coal[3], h_output_coal[4]);
    printf("Uncoalesced: %.2f %.2f %.2f %.2f %.2f\n", 
           h_output_uncoal[0], h_output_uncoal[1], h_output_uncoal[2], h_output_uncoal[3], h_output_uncoal[4]);
    
    // Cleanup
    free(h_input); free(h_output_uncoal); free(h_output_coal); free(h_output_student);
    cudaFree(d_input); cudaFree(d_output_uncoal); cudaFree(d_output_coal); cudaFree(d_output_student);
    
    printf("\nExercise completed! Notice the performance difference between coalesced and uncoalesced access.\n");
    
    return 0;
}