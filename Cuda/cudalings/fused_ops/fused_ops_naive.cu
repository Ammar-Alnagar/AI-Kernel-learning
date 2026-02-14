#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>

// Naive fused operation: fused multiply-add (FMA) - A * B + C
__global__ void fused_multiply_add_naive(float *A, float *B, float *C, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        result[idx] = A[idx] * B[idx] + C[idx];
    }
}

// More complex fused operation: fused multiply-add with activation (ReLU)
__global__ void fused_multiply_add_relu_naive(float *A, float *B, float *C, float *result, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        float temp = A[idx] * B[idx] + C[idx];
        result[idx] = (temp > 0.0f) ? temp : 0.0f;  // ReLU activation
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

// CPU equivalent for verification
void cpu_fma(float *A, float *B, float *C, float *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = A[i] * B[i] + C[i];
    }
}

int main(int argc, char **argv) {
    int n = 1 << 20; // 1M elements by default
    if (argc > 1) {
        n = atoi(argv[1]);
    }
    
    printf("Fused multiply-add of %d elements\n", n);
    
    size_t input_size = n * sizeof(float);
    
    // Host arrays
    float *h_A = (float*)malloc(input_size);
    float *h_B = (float*)malloc(input_size);
    float *h_C = (float*)malloc(input_size);
    float *h_result = (float*)malloc(input_size);
    float *h_expected = (float*)malloc(input_size);
    
    // Initialize with random values
    for (int i = 0; i < n; i++) {
        h_A[i] = (float)(rand()) / RAND_MAX;
        h_B[i] = (float)(rand()) / RAND_MAX;
        h_C[i] = (float)(rand()) / RAND_MAX;
    }
    
    // Calculate expected result on CPU
    cpu_fma(h_A, h_B, h_C, h_expected, n);
    
    // Device arrays
    float *d_A, *d_B, *d_C, *d_result;
    checkCudaError(cudaMalloc(&d_A, input_size), "cudaMalloc d_A");
    checkCudaError(cudaMalloc(&d_B, input_size), "cudaMalloc d_B");
    checkCudaError(cudaMalloc(&d_C, input_size), "cudaMalloc d_C");
    checkCudaError(cudaMalloc(&d_result, input_size), "cudaMalloc d_result");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_A, h_A, input_size, cudaMemcpyHostToDevice), "cudaMemcpy d_A");
    checkCudaError(cudaMemcpy(d_B, h_B, input_size, cudaMemcpyHostToDevice), "cudaMemcpy d_B");
    checkCudaError(cudaMemcpy(d_C, h_C, input_size, cudaMemcpyHostToDevice), "cudaMemcpy d_C");
    
    // Kernel configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    
    printf("Launching kernel with %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    fused_multiply_add_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_result, n);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    fused_multiply_add_naive<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, d_result, n);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_result, d_result, input_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_result");
    
    // Verification: Check a few random points
    bool success = true;
    int num_checks = min(1000, n);
    
    for (int check = 0; check < num_checks; check++) {
        int idx = rand() % n;
        
        if (abs(h_result[idx] - h_expected[idx]) > 1e-5) {
            printf("Verification failed at index %d: expected %f, got %f\n", idx, h_expected[idx], h_result[idx]);
            success = false;
            break;
        }
    }
    
    if (success) {
        printf("Verification PASSED (checked %d random elements)\n", num_checks);
        
        // Calculate performance metrics
        double flops = 2.0 * n; // 1 multiply + 1 add per element
        double gflops = (flops / 1e9) / (kernel_time_ms / 1000.0);
        double bytes_moved = 4.0 * n * sizeof(float); // 3 reads + 1 write
        double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_A); free(h_B); free(h_C); free(h_result); free(h_expected);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_result);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}