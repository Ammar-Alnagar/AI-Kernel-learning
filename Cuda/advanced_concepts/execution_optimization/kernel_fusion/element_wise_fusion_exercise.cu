/*
 * Element-Wise Fusion (e.g., GELU + Add) Exercise
 *
 * This exercise demonstrates how to fuse element-wise operations to reduce memory traffic.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Constants for GELU approximation
#define GELU_SCALAR 1.702f

// Kernel 1: Unfused Operations (Separate Kernels)
__global__ void geluKernel(float* input, float* temp, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // GELU(x) = x * sigmoid(1.702 * x)
        float x = input[tid];
        float sigmoid_val = 1.0f / (1.0f + expf(-GELU_SCALAR * x));
        temp[tid] = x * sigmoid_val;
    }
}

__global__ void addKernel(float* input1, float* input2, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        output[tid] = input1[tid] + input2[tid];
    }
}

// Kernel 2: Fused GELU + Add (Single Kernel)
__global__ void fusedGeluAdd(float* input, float* residual, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Compute GELU(input)
        float x = input[tid];
        float sigmoid_val = 1.0f / (1.0f + expf(-GELU_SCALAR * x));
        float gelu_result = x * sigmoid_val;
        
        // Add residual connection
        output[tid] = gelu_result + residual[tid];
    }
}

// Kernel 3: Alternative Fused Operation (Add + GELU)
__global__ void fusedAddGelu(float* input, float* residual, float* output, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n) {
        // Add first
        float sum = input[tid] + residual[tid];
        
        // Then apply GELU
        float sigmoid_val = 1.0f / (1.0f + expf(-GELU_SCALAR * sum));
        output[tid] = sum * sigmoid_val;
    }
}

// Kernel 4: Student Exercise - Implement fused operation with multiple activations
__global__ void studentFusedMultiActivation(float* input1, float* input2, float* input3, 
                                         float* output, int n, int activation_type) {
    // TODO: Implement a fused operation that combines multiple inputs with different activations
    // HINT: Support different activation types (GELU, ReLU, SiLU, etc.) in a single kernel
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // FIX: Combine multiple inputs based on activation_type
        float result = 0.0f;
        
        if (activation_type == 0) {  // GELU
            // Combine inputs and apply GELU
            float x = input1[tid] + input2[tid] + input3[tid];
            float sigmoid_val = 1.0f / (1.0f + expf(-GELU_SCALAR * x));
            result = x * sigmoid_val;
        } else if (activation_type == 1) {  // ReLU
            // Combine inputs and apply ReLU
            float x = input1[tid] + input2[tid] + input3[tid];
            result = fmaxf(0.0f, x);
        } else if (activation_type == 2) {  // SiLU (Swish)
            // Combine inputs and apply SiLU: x * sigmoid(x)
            float x = input1[tid] + input2[tid] + input3[tid];
            float sigmoid_val = 1.0f / (1.0f + expf(-x));
            result = x * sigmoid_val;
        }
        // ADD MORE ACTIVATION TYPES AS NEEDED
        
        output[tid] = result;
    }
}

// Kernel 5: Student Exercise - Implement fused operation with bias and activation
__global__ void studentFusedBiasActivation(float* input, float* bias, float* skip_connection, 
                                        float* output, int n, float alpha, float beta) {
    // TODO: Implement fused operation: output = alpha * activation(input + bias) + beta * skip_connection
    // HINT: Combine bias addition, activation function, and residual connection in one kernel
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // FIX: Implement the complete fused operation
        // Step 1: Add bias to input
        float biased_input = /* INPUT + BIAS */;
        
        // Step 2: Apply activation function (GELU)
        float sigmoid_val = 1.0f / (1.0f + expf(-GELU_SCALAR * biased_input));
        float activated = biased_input * sigmoid_val;
        
        // Step 3: Scale activated result and add scaled skip connection
        output[tid] = alpha * activated + beta * skip_connection[tid];
    }
}

// Kernel 6: Student Exercise - Implement complex element-wise fusion
__global__ void studentComplexFusion(float* A, float* B, float* C, float* D, float* output, 
                                   int n, float* params) {
    // TODO: Implement a complex fusion of multiple element-wise operations
    // Example: output = GELU(A * B + C) + D * params[0] + params[1]
    // HINT: Chain multiple operations in a single kernel to maximize efficiency
    
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid < n) {
        // FIX: Implement the complex fusion operation
        // Step 1: Multiply A and B
        float mult_result = A[tid] * B[tid];
        
        // Step 2: Add C
        float add_result = mult_result + C[tid];
        
        // Step 3: Apply GELU
        float sigmoid_val = 1.0f / (1.0f + expf(-GELU_SCALAR * add_result));
        float gelu_result = add_result * sigmoid_val;
        
        // Step 4: Add scaled D and bias
        output[tid] = gelu_result + D[tid] * params[0] + params[1];
    }
}

// Utility function to initialize array
void initArray(float* arr, int n, float start_val = 0.0f) {
    for (int i = 0; i < n; i++) {
        arr[i] = start_val + (i % 20 - 10) * 0.1f;  // Mix of positive/negative values
    }
}

int main() {
    printf("=== Element-Wise Fusion (GELU + Add) Exercise ===\n");
    printf("Learn to implement fused element-wise operations for improved performance.\n\n");

    // Setup parameters
    const int N = 1024 * 16;  // Multiple of block size
    size_t bytes = N * sizeof(float);
    
    // Allocate host memory
    float *h_input, *h_residual, *h_input2, *h_input3, *h_bias, *h_skip, *h_params;
    float *h_output_unfused, *h_output_fused, *h_output_multi, *h_output_bias, *h_output_complex;
    
    h_input = (float*)malloc(bytes);
    h_residual = (float*)malloc(bytes);
    h_input2 = (float*)malloc(bytes);
    h_input3 = (float*)malloc(bytes);
    h_bias = (float*)malloc(bytes);
    h_skip = (float*)malloc(bytes);
    h_params = (float*)malloc(2 * sizeof(float));  // For complex fusion parameters
    
    h_output_unfused = (float*)malloc(bytes);
    h_output_fused = (float*)malloc(bytes);
    h_output_multi = (float*)malloc(bytes);
    h_output_bias = (float*)malloc(bytes);
    h_output_complex = (float*)malloc(bytes);
    
    // Initialize data
    initArray(h_input, N, 0.1f);
    initArray(h_residual, N, 0.05f);
    initArray(h_input2, N, -0.05f);
    initArray(h_input3, N, 0.15f);
    initArray(h_bias, N, 0.02f);
    initArray(h_skip, N, 0.08f);
    h_params[0] = 0.5f;  // Scale factor
    h_params[1] = 0.1f;  // Bias
    
    // Initialize output arrays to zero
    memset(h_output_unfused, 0, bytes);
    memset(h_output_fused, 0, bytes);
    memset(h_output_multi, 0, bytes);
    memset(h_output_bias, 0, bytes);
    memset(h_output_complex, 0, bytes);
    
    // Allocate device memory
    float *d_input, *d_residual, *d_input2, *d_input3, *d_bias, *d_skip, *d_params;
    float *d_temp, *d_output_unfused, *d_output_fused, *d_output_multi, *d_output_bias, *d_output_complex;
    
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_residual, bytes);
    cudaMalloc(&d_input2, bytes);
    cudaMalloc(&d_input3, bytes);
    cudaMalloc(&d_bias, bytes);
    cudaMalloc(&d_skip, bytes);
    cudaMalloc(&d_params, 2 * sizeof(float));
    
    cudaMalloc(&d_temp, bytes);
    cudaMalloc(&d_output_unfused, bytes);
    cudaMalloc(&d_output_fused, bytes);
    cudaMalloc(&d_output_multi, bytes);
    cudaMalloc(&d_output_bias, bytes);
    cudaMalloc(&d_output_complex, bytes);
    
    // Copy data to device
    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_residual, h_residual, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input2, h_input2, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_input3, h_input3, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_skip, h_skip, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_params, h_params, 2 * sizeof(float), cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (N + blockSize - 1) / blockSize;
    
    // Run unfused operations (two kernels)
    printf("Running unfused GELU + Add operations (two kernels)...\n");
    geluKernel<<<gridSize, blockSize>>>(d_input, d_temp, N);
    cudaDeviceSynchronize();
    addKernel<<<gridSize, blockSize>>>(d_temp, d_residual, d_output_unfused, N);
    cudaDeviceSynchronize();
    
    // Run fused GELU + Add kernel
    printf("Running fused GELU + Add kernel...\n");
    fusedGeluAdd<<<gridSize, blockSize>>>(d_input, d_residual, d_output_fused, N);
    cudaDeviceSynchronize();
    
    // Run fused Add + GELU kernel
    printf("Running fused Add + GELU kernel...\n");
    fusedAddGelu<<<gridSize, blockSize>>>(d_input, d_residual, d_output_fused, N);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student fusion exercises (complete the code first!)...\n");
    
    // Multi-activation fusion exercise
    studentFusedMultiActivation<<<gridSize, blockSize>>>(d_input, d_input2, d_input3, 
                                                       d_output_multi, N, 0);  // GELU
    cudaDeviceSynchronize();
    
    // Bias + activation + skip fusion exercise
    studentFusedBiasActivation<<<gridSize, blockSize>>>(d_input, d_bias, d_skip, 
                                                       d_output_bias, N, 1.0f, 1.0f);
    cudaDeviceSynchronize();
    
    // Complex fusion exercise
    studentComplexFusion<<<gridSize, blockSize>>>(d_input, d_residual, d_input2, d_input3, 
                                                 d_output_complex, N, d_params);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the fusion implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_unfused, d_output_unfused, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused, d_output_fused, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_multi, d_output_multi, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_bias, d_output_bias, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_complex, d_output_complex, bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements):\n");
    printf("Input:     %.3f %.3f %.3f %.3f %.3f\n", 
           h_input[0], h_input[1], h_input[2], h_input[3], h_input[4]);
    printf("Unfused:   %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_unfused[0], h_output_unfused[1], h_output_unfused[2], h_output_unfused[3], h_output_unfused[4]);
    printf("Fused:     %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_fused[0], h_output_fused[1], h_output_fused[2], h_output_fused[3], h_output_fused[4]);
    printf("Multi-fuse: %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_multi[0], h_output_multi[1], h_output_multi[2], h_output_multi[3], h_output_multi[4]);
    
    // Cleanup
    free(h_input); free(h_residual); free(h_input2); free(h_input3); 
    free(h_bias); free(h_skip); free(h_params);
    free(h_output_unfused); free(h_output_fused); free(h_output_multi); 
    free(h_output_bias); free(h_output_complex);
    
    cudaFree(d_input); cudaFree(d_residual); cudaFree(d_input2); cudaFree(d_input3);
    cudaFree(d_bias); cudaFree(d_skip); cudaFree(d_params);
    cudaFree(d_temp); cudaFree(d_output_unfused); cudaFree(d_output_fused);
    cudaFree(d_output_multi); cudaFree(d_output_bias); cudaFree(d_output_complex);
    
    printf("\nExercise completed! Notice how element-wise fusion reduces memory traffic.\n");
    
    return 0;
}