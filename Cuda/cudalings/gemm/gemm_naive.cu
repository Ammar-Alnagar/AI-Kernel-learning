#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive GEMM kernel: C = alpha * A * B + beta * C
__global__ void gemm_naive(float alpha, float *A, float *B, float beta, float *C, 
                          int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

// Utility function to check CUDA errors
void checkCudaError(cudaError_t error, const char* function) {
    if (error != cudaSuccess) {
        fprintf(stderr, "%s: %s\n", function, cudaGetErrorString(error));
        exit(1);
    }
}

// Timing utility
double get_time_diff(struct timeval start, struct timeval end) {
    return (end.tv_sec - start.tv_sec) + (end.tv_usec - start.tv_usec) / 1000000.0;
}

// CPU GEMM for verification
void cpu_gemm(float alpha, float *A, float *B, float beta, float *C, 
              float *result, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            result[i * N + j] = alpha * sum + beta * C[i * N + j];
        }
    }
}

int main(int argc, char **argv) {
    int M = 512, N = 512, K = 512; // Default matrix dimensions
    float alpha = 1.0f, beta = 0.0f; // Standard GEMM: C = A * B
    
    if (argc >= 4) {
        M = atoi(argv[1]);
        N = atoi(argv[2]);
        K = atoi(argv[3]);
    }
    if (argc >= 5) {
        alpha = atof(argv[4]);
    }
    if (argc >= 6) {
        beta = atof(argv[5]);
    }
    
    printf("GEMM: C(%dx%d) = %f * A(%dx%d) * B(%dx%d) + %f * C\n", 
           M, N, alpha, M, K, K, N, beta);
    
    size_t sizeA = M * K * sizeof(float);
    size_t sizeB = K * N * sizeof(float);
    size_t sizeC = M * N * sizeof(float);
    
    // Host arrays
    float *h_A = (float*)malloc(sizeA);
    float *h_B = (float*)malloc(sizeB);
    float *h_C = (float*)malloc(sizeC);
    float *h_result = (float*)malloc(sizeC);
    
    // Initialize host arrays with random values
    for (int i = 0; i < M * K; i++) {
        h_A[i] = ((float)(rand() % 2000)) / 1000.0f - 1.0f; // Range [-1, 1]
    }
    for (int i = 0; i < K * N; i++) {
        h_B[i] = ((float)(rand() % 2000)) / 1000.0f - 1.0f; // Range [-1, 1]
    }
    for (int i = 0; i < M * N; i++) {
        h_C[i] = ((float)(rand() % 2000)) / 1000.0f - 1.0f; // Range [-1, 1]
    }
    
    // Calculate expected result on CPU
    cpu_gemm(alpha, h_A, h_B, beta, h_C, h_result, M, N, K);
    
    // Device arrays
    float *d_A, *d_B, *d_C;
    checkCudaError(cudaMalloc(&d_A, sizeA), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, sizeB), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, sizeC), "cudaMalloc d_C");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A, sizeA, cudaMemcpyHostToDevice), "cudaMemcpy h_A to d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, sizeB, cudaMemcpyHostToDevice), "cudaMemcpy h_B to d_B");
    checkCudaError(cudaMemcpy(d_C, h_C, sizeC, cudaMemcpyHostToDevice), "cudaMemcpy h_C to d_C");
    
    // Kernel configuration
    dim3 blockSize(16, 16);  // 256 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    
    printf("Launching kernel with grid (%d,%d) and block (%d,%d)\n", 
           gridSize.x, gridSize.y, blockSize.x, blockSize.y);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    gemm_naive<<<gridSize, blockSize>>>(alpha, d_A, d_B, beta, d_C, M, N, K);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    gemm_naive<<<gridSize, blockSize>>>(alpha, d_A, d_B, beta, d_C, M, N, K);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_C, d_C, sizeC, cudaMemcpyDeviceToHost), "cudaMemcpy d_C to h_C");
    
    // Verification: Check a few random elements
    bool success = true;
    int num_checks = min(100, M*N);
    
    for (int check = 0; check < num_checks; check++) {
        int i = rand() % M;
        int j = rand() % N;
        
        if (abs(h_C[i * N + j] - h_result[i * N + j]) > 1e-2) {  // Increased tolerance due to floating point precision
            printf("Verification failed at [%d][%d]: expected %f, got %f\n", 
                   i, j, h_result[i * N + j], h_C[i * N + j]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED (checked %d random elements)\n", num_checks);
        
        // Calculate performance metrics
        double flops = 2.0 * M * N * K; // 2 operations per element (multiply + add)
        double gflops = flops / (1e9 * kernel_time_ms / 1000.0);
        double bytes_moved = (M*K + K*N + M*N) * sizeof(float); // Input + output
        double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_result);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}