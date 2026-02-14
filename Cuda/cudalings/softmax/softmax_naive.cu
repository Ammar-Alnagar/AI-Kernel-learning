#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <sys/time.h>
#include <math.h>

// Naive softmax kernel - one thread per row
__global__ void softmax_naive(float *input, float *output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Find max value in the row for numerical stability
        float max_val = input[row * cols];
        for (int j = 1; j < cols; j++) {
            if (input[row * cols + j] > max_val) {
                max_val = input[row * cols + j];
            }
        }
        
        // Compute sum of exponentials
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(input[row * cols + j] - max_val);
        }
        
        // Compute softmax values
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - max_val) / sum;
        }
    }
}

// Optimized softmax with shared memory
__global__ void softmax_shared_mem(float *input, float *output, int rows, int cols) {
    extern __shared__ float temp[];
    
    int row = blockIdx.x;
    int tid = threadIdx.x;
    
    if (row < rows) {
        // Load data into shared memory
        if (tid < cols) {
            temp[tid] = input[row * cols + tid];
        }
        __syncthreads();
        
        if (tid == 0) {
            // Find max value in the row
            float max_val = temp[0];
            for (int j = 1; j < cols; j++) {
                if (temp[j] > max_val) {
                    max_val = temp[j];
                }
            }
            
            // Compute sum of exponentials
            float sum = 0.0f;
            for (int j = 0; j < cols; j++) {
                temp[j] = expf(temp[j] - max_val);
                sum += temp[j];
            }
            
            // Normalize and write back
            for (int j = 0; j < cols; j++) {
                output[row * cols + j] = temp[j] / sum;
            }
        }
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

// CPU softmax for verification
void cpu_softmax(float *input, float *output, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        // Find max value in the row
        float max_val = input[i * cols];
        for (int j = 1; j < cols; j++) {
            if (input[i * cols + j] > max_val) {
                max_val = input[i * cols + j];
            }
        }
        
        // Compute sum of exponentials
        float sum = 0.0f;
        for (int j = 0; j < cols; j++) {
            sum += expf(input[i * cols + j] - max_val);
        }
        
        // Compute softmax values
        for (int j = 0; j < cols; j++) {
            output[i * cols + j] = expf(input[i * cols + j] - max_val) / sum;
        }
    }
}

// Verify that the output is a valid probability distribution (sums to 1)
bool verify_softmax_row(float *row, int cols, float tolerance = 1e-4) {
    float sum = 0.0f;
    for (int j = 0; j < cols; j++) {
        sum += row[j];
    }
    return fabsf(sum - 1.0f) < tolerance;
}

int main(int argc, char **argv) {
    int rows = 128, cols = 512; // Default dimensions
    
    if (argc >= 3) {
        rows = atoi(argv[1]);
        cols = atoi(argv[2]);
    }
    
    printf("Softmax: %d rows with %d columns each\n", rows, cols);
    
    size_t input_size = rows * cols * sizeof(float);
    
    // Host arrays
    float *h_input = (float*)malloc(input_size);
    float *h_output = (float*)malloc(input_size);
    float *h_expected = (float*)malloc(input_size);
    
    // Initialize with random values
    for (int i = 0; i < rows * cols; i++) {
        h_input[i] = ((float)(rand() % 2000)) / 1000.0f - 1.0f; // Range [-1, 1]
    }
    
    // Calculate expected result on CPU
    cpu_softmax(h_input, h_expected, rows, cols);
    
    // Device arrays
    float *d_input, *d_output;
    checkCudaError(cudaMalloc(&d_input, input_size), "cudaMalloc d_input");
    checkCudaError(cudaMalloc(&d_output, input_size), "cudaMalloc d_output");
    
    // Copy data to device
    checkCudaError(cudaMemcpy(d_input, h_input, input_size, cudaMemcpyHostToDevice), "cudaMemcpy d_input");
    
    // Kernel configuration
    int threadsPerBlock = min(cols, 1024);  // Use enough threads for the largest row
    int blocksPerGrid = rows;
    
    printf("Launching kernel with %d blocks of %d threads each\n", blocksPerGrid, threadsPerBlock);
    
    // Timing variables
    struct timeval start, end;
    cudaEvent_t start_event, stop_event;
    
    checkCudaError(cudaEventCreate(&start_event), "cudaEventCreate");
    checkCudaError(cudaEventCreate(&stop_event), "cudaEventCreate");
    
    // Warm up run
    softmax_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    cudaDeviceSynchronize();
    
    // Timing the kernel execution
    gettimeofday(&start, NULL);
    cudaEventRecord(start_event);
    
    softmax_naive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, rows, cols);
    
    cudaEventRecord(stop_event);
    cudaEventSynchronize(stop_event);
    gettimeofday(&end, NULL);
    
    float kernel_time_ms;
    cudaEventElapsedTime(&kernel_time_ms, start_event, stop_event);
    double total_time_s = get_time_diff(start, end);
    
    // Copy result back to host
    checkCudaError(cudaMemcpy(h_output, d_output, input_size, cudaMemcpyDeviceToHost), "cudaMemcpy d_output");
    
    // Verification: Check a few random rows
    bool success = true;
    int num_checks = min(10, rows);
    
    for (int check = 0; check < num_checks; check++) {
        int row_idx = (check * rows) / num_checks;
        
        // Verify this row sums to 1
        if (!verify_softmax_row(&h_output[row_idx * cols], cols)) {
            printf("Row %d does not sum to 1\n", row_idx);
            success = false;
            break;
        }
        
        // Compare with CPU result
        for (int j = 0; j < cols; j++) {
            if (fabsf(h_output[row_idx * cols + j] - h_expected[row_idx * cols + j]) > 1e-4) {
                printf("Verification failed at row %d, col %d: expected %f, got %f\n", 
                       row_idx, j, h_expected[row_idx * cols + j], h_output[row_idx * cols + j]);
                success = false;
                break;
            }
        }
        
        if (!success) break;
    }
    
    if (success) {
        printf("Verification PASSED (checked %d rows)\n", num_checks);
        
        // Calculate performance metrics
        double flops = rows * cols * (2.0 + 1.0); // exp + division per element, plus operations for max and sum
        double gflops = (flops / 1e9) / (kernel_time_ms / 1000.0);
        double bytes_moved = 2.0 * rows * cols * sizeof(float); // Read + write
        double bandwidth_gbs = (bytes_moved / (1024.0 * 1024.0 * 1024.0)) / (kernel_time_ms / 1000.0);
        
        printf("Kernel execution time: %.3f ms\n", kernel_time_ms);
        printf("Total execution time: %.3f s\n", total_time_s);
        printf("Performance: %.2f GFLOPS\n", gflops);
        printf("Effective bandwidth: %.2f GB/s\n", bandwidth_gbs);
    } else {
        printf("Verification FAILED\n");
    }
    
    // Cleanup
    free(h_input); free(h_output); free(h_expected);
    cudaFree(d_input); cudaFree(d_output);
    cudaEventDestroy(start_event); cudaEventDestroy(stop_event);
    
    return success ? 0 : 1;
}