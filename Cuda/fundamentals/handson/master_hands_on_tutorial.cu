/*
 * Master CUDA Hands-On Tutorial
 *
 * This tutorial combines all the individual hands-on exercises into one comprehensive
 * program that demonstrates all the concepts practiced in the separate exercises.
 * Students can use this to practice all concepts together or individually.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Include all the kernels from individual exercises
// Vector Addition Kernel
__global__ void vectorAdd(float* A, float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        C[i] = A[i] + B[i];
    }
}

// Matrix Multiplication Kernel
__global__ void matrixMul(float* A, float* B, float* C, int width) {
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

// Reduction Kernel
__global__ void reductionSum(float* input, float* output, int n) {
    extern __shared__ float sdata[];
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        sdata[tid] = input[i];
    } else {
        sdata[tid] = 0.0f;
    }
    __syncthreads();

    for (int s = 1; s < blockDim.x; s *= 2) {
        if (tid + s < blockDim.x) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

// Coalesced Memory Access Kernel
__global__ void coalescedCopy(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input[tid];
    }
}

// Shared Memory Transpose Kernel (with bank conflict fix)
__global__ void sharedMemoryTranspose(float* input, float* output, int width) {
    __shared__ float tile[32][33];  // Padded to avoid bank conflicts

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        tile[threadIdx.y][threadIdx.x] = input[y * width + x];
    }
    __syncthreads();
    
    x = blockIdx.y * 32 + threadIdx.x;
    y = blockIdx.x * 32 + threadIdx.y;
    
    if (x < width && y < width) {
        output[y * width + x] = tile[threadIdx.x][threadIdx.y + threadIdx.x/32]; // Fixed access
    }
}

// Atomic Operations Kernel
__global__ void atomicHistogram(unsigned char* input, unsigned int* histogram, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        unsigned char value = input[tid];
        atomicAdd(&histogram[value], 1);
    }
}

// Warp Shuffle Kernel
__global__ void warpShuffleExample(float* input, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int laneId = threadIdx.x % 32;
    
    if (tid < n) {
        float value = input[tid];
        float neighborValue = __shfl_down_sync(0xFFFFFFFF, value, 1, 32);
        output[tid] = neighborValue;
    }
}

// Simple computation kernel for streams
__global__ void asyncComputation(float* data, int n, float factor) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        data[tid] = data[tid] * factor + 1.0f;
    }
}

// Utility function to initialize vectors
void initVector(float* vec, int n, float start_val = 1.0f) {
    for (int i = 0; i < n; i++) {
        vec[i] = start_val + i * 0.1f;
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int width, float start_val = 1.0f) {
    for (int i = 0; i < width * width; i++) {
        mat[i] = start_val + i * 0.01f;
    }
}

// Utility function to initialize byte array
void initByteArray(unsigned char* arr, int n) {
    for (int i = 0; i < n; i++) {
        arr[i] = i % 256;
    }
}

// Utility function to print first few elements
void printVector(float* vec, int n, int count = 10) {
    printf("First %d elements: ", count > n ? n : count);
    for (int i = 0; i < (count > n ? n : count); i++) {
        printf("%.2f ", vec[i]);
    }
    printf("\n");
}

int main() {
    printf("=== Master CUDA Hands-On Tutorial ===\n");
    printf("This tutorial demonstrates all the concepts from individual exercises.\n\n");

    // Exercise 1: Vector Addition
    printf("1. Vector Addition Exercise:\n");
    const int N1 = 1024;
    size_t size1 = N1 * sizeof(float);
    
    float *h_A, *h_B, *h_C;
    h_A = (float*)malloc(size1);
    h_B = (float*)malloc(size1);
    h_C = (float*)malloc(size1);
    
    initVector(h_A, N1, 1.0f);
    initVector(h_B, N1, 2.0f);
    
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, size1);
    cudaMalloc(&d_B, size1);
    cudaMalloc(&d_C, size1);
    
    cudaMemcpy(d_A, h_A, size1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size1, cudaMemcpyHostToDevice);
    
    int blockSize1 = 256;
    int gridSize1 = (N1 + blockSize1 - 1) / blockSize1;
    
    vectorAdd<<<gridSize1, blockSize1>>>(d_A, d_B, d_C, N1);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C, d_C, size1, cudaMemcpyDeviceToHost);
    printVector(h_C, N1, 5);
    
    // Cleanup for vector addition
    free(h_A); free(h_B); free(h_C);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    
    printf("\n2. Matrix Multiplication Exercise:\n");
    const int WIDTH2 = 32;
    const int N2 = WIDTH2 * WIDTH2;
    size_t size2 = N2 * sizeof(float);
    
    float *h_A2, *h_B2, *h_C2;
    h_A2 = (float*)malloc(size2);
    h_B2 = (float*)malloc(size2);
    h_C2 = (float*)malloc(size2);
    
    initMatrix(h_A2, WIDTH2, 1.0f);
    initMatrix(h_B2, WIDTH2, 2.0f);
    
    float *d_A2, *d_B2, *d_C2;
    cudaMalloc(&d_A2, size2);
    cudaMalloc(&d_B2, size2);
    cudaMalloc(&d_C2, size2);
    
    cudaMemcpy(d_A2, h_A2, size2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B2, h_B2, size2, cudaMemcpyHostToDevice);
    
    dim3 blockSize2(16, 16);
    dim3 gridSize2((WIDTH2 + blockSize2.x - 1) / blockSize2.x, 
                   (WIDTH2 + blockSize2.y - 1) / blockSize2.y);
    
    matrixMul<<<gridSize2, blockSize2>>>(d_A2, d_B2, d_C2, WIDTH2);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_C2, d_C2, size2, cudaMemcpyDeviceToHost);
    printf("Matrix multiplication completed.\n");
    
    // Cleanup for matrix multiplication
    free(h_A2); free(h_B2); free(h_C2);
    cudaFree(d_A2); cudaFree(d_B2); cudaFree(d_C2);
    
    printf("\n3. Reduction Operation Exercise:\n");
    const int N3 = 1024;
    size_t size3 = N3 * sizeof(float);
    size_t blockSize3 = 256;
    size_t gridSize3 = (N3 + blockSize3 - 1) / blockSize3;
    
    float *h_input3, *h_output3;
    h_input3 = (float*)malloc(size3);
    h_output3 = (float*)malloc(gridSize3 * sizeof(float));
    
    initVector(h_input3, N3, 1.0f);
    
    float *d_input3, *d_output3;
    cudaMalloc(&d_input3, size3);
    cudaMalloc(&d_output3, gridSize3 * sizeof(float));
    
    cudaMemcpy(d_input3, h_input3, size3, cudaMemcpyHostToDevice);
    
    reductionSum<<<gridSize3, blockSize3, blockSize3*sizeof(float)>>>(d_input3, d_output3, N3);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_output3, d_output3, gridSize3 * sizeof(float), cudaMemcpyDeviceToHost);
    
    float final_sum = 0.0f;
    for (int i = 0; i < gridSize3; i++) {
        final_sum += h_output3[i];
    }
    printf("Reduction sum: %.2f\n", final_sum);
    
    // Cleanup for reduction
    free(h_input3); free(h_output3);
    cudaFree(d_input3); cudaFree(d_output3);
    
    printf("\n4. Atomic Operations Exercise:\n");
    const int N4 = 10000;
    const int HIST_SIZE = 256;
    
    size_t input_size4 = N4 * sizeof(unsigned char);
    size_t hist_size4 = HIST_SIZE * sizeof(unsigned int);
    
    unsigned char *h_input4;
    unsigned int *h_histogram4;
    
    h_input4 = (unsigned char*)malloc(input_size4);
    h_histogram4 = (unsigned int*)calloc(HIST_SIZE, sizeof(unsigned int));
    
    initByteArray(h_input4, N4);
    
    unsigned char *d_input4;
    unsigned int *d_histogram4;
    
    cudaMalloc(&d_input4, input_size4);
    cudaMalloc(&d_histogram4, hist_size4);
    
    cudaMemset(d_histogram4, 0, hist_size4);
    cudaMemcpy(d_input4, h_input4, input_size4, cudaMemcpyHostToDevice);
    
    int blockSize4 = 256;
    int gridSize4 = (N4 + blockSize4 - 1) / blockSize4;
    
    atomicHistogram<<<gridSize4, blockSize4>>>(d_input4, d_histogram4, N4);
    cudaDeviceSynchronize();
    
    cudaMemcpy(h_histogram4, d_histogram4, hist_size4, cudaMemcpyDeviceToHost);
    printf("Atomic histogram - First 5 bins: %u %u %u %u %u\n", 
           h_histogram4[0], h_histogram4[1], h_histogram4[2], h_histogram4[3], h_histogram4[4]);
    
    // Cleanup for atomic operations
    free(h_input4); free(h_histogram4);
    cudaFree(d_input4); cudaFree(d_histogram4);
    
    printf("\nAll exercises completed successfully!\n");
    printf("Refer to individual exercise files for hands-on practice with incomplete code.\n");
    
    return 0;
}