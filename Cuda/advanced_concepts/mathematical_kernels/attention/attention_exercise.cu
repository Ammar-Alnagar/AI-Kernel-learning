/*
 * Attention Mechanism Exercise
 *
 * This exercise demonstrates how to implement efficient attention mechanisms,
 * including naive, fused, and FlashAttention-style implementations.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Kernel 1: Naive Attention (INEFFICIENT)
__global__ void naiveAttention(float* Q, float* K, float* V, float* output, 
                              int batch_size, int seq_len, int head_dim) {
    int batch_id = blockIdx.x;
    int head_id = blockIdx.y;
    int row = threadIdx.y;  // Query position
    int col = threadIdx.x;  // Key position
    
    if (batch_id < batch_size && head_id < 1) {  // Simplified for single head
        // Compute attention scores
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += Q[batch_id * seq_len * head_dim + row * head_dim + d] * 
                     K[batch_id * seq_len * head_dim + col * head_dim + d];
        }
        score = score / sqrtf((float)head_dim);  // Scale
        
        // Apply softmax implicitly in the next step
        // For simplicity, we'll just compute the score here
        // A full implementation would compute softmax over all keys for each query
    }
}

// Kernel 2: Fused Attention (More Efficient)
__global__ void fusedAttention(float* Q, float* K, float* V, float* output,
                              int batch_size, int seq_len, int head_dim) {
    int batch_id = blockIdx.x;
    int query_pos = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_id < batch_size && query_pos < seq_len) {
        // Compute query vector
        float q_vec[64];  // Assuming max head_dim of 64
        for (int d = 0; d < head_dim; d++) {
            q_vec[d] = Q[batch_id * seq_len * head_dim + query_pos * head_dim + d];
        }
        
        // Compute attention weights and output in one pass
        float weights[512];  // Assuming max seq_len of 512
        float max_weight = -INFINITY;
        float weight_sum = 0.0f;
        
        // Compute raw attention scores and find max for numerical stability
        for (int k = 0; k < seq_len; k++) {
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_vec[d] * 
                         K[batch_id * seq_len * head_dim + k * head_dim + d];
            }
            score = score / sqrtf((float)head_dim);
            weights[k] = score;
            if (score > max_weight) max_weight = score;
        }
        
        // Compute normalized weights (softmax)
        for (int k = 0; k < seq_len; k++) {
            weights[k] = expf(weights[k] - max_weight);
            weight_sum += weights[k];
        }
        for (int k = 0; k < seq_len; k++) {
            weights[k] = weights[k] / weight_sum;
        }
        
        // Compute output as weighted sum of values
        for (int d = 0; d < head_dim; d++) {
            float out_val = 0.0f;
            for (int k = 0; k < seq_len; k++) {
                out_val += weights[k] * 
                          V[batch_id * seq_len * head_dim + k * head_dim + d];
            }
            output[batch_id * seq_len * head_dim + query_pos * head_dim + d] = out_val;
        }
    }
}

// Kernel 3: Student Exercise - Implement FlashAttention-style block-wise attention
__global__ void studentFlashAttention(float* Q, float* K, float* V, float* output,
                                     int batch_size, int seq_len, int head_dim) {
    // TODO: Implement a simplified FlashAttention-style algorithm
    // HINT: Process attention computation in blocks to reduce memory usage
    
    int batch_id = blockIdx.x;
    int query_block_start = blockIdx.y;
    int key_value_block_start = blockIdx.z;
    
    // FIX: Implement block-wise attention computation
    // Process queries and keys in blocks to reduce memory footprint
    // Use online softmax computation to avoid materializing full attention matrix
    
    // Define block size for processing
    const int BLOCK_SIZE = 64;  // Process 64 tokens at a time
    
    // FIX: Implement the core algorithm:
    // 1. Load a block of Q, K, V matrices into shared memory
    // 2. Compute partial attention scores and values
    // 3. Combine results from different blocks
    
    // Example structure (you'll need to implement the full logic):
    /*
    // Shared memory for blocks of Q, K, V
    extern __shared__ float shared_mem[];
    float *Q_block = shared_mem;
    float *K_block = Q_block + BLOCK_SIZE * head_dim;
    float *V_block = K_block + BLOCK_SIZE * head_dim;
    float *O_block = V_block + BLOCK_SIZE * head_dim;
    float *L_block = O_block + BLOCK_SIZE * head_dim;  // Normalization terms
    float *M_block = L_block + BLOCK_SIZE;             // Max values
    
    // Load blocks of K and V into shared memory
    // ...
    
    // Iterate through query blocks
    for (int q_start = query_block_start * BLOCK_SIZE; 
         q_start < min((query_block_start + 1) * BLOCK_SIZE, seq_len); 
         q_start++) {
        
        // Load query block
        // ...
        
        // Compute attention for this query block with current key/value block
        // Use online softmax algorithm to avoid materializing full attention matrix
        // ...
    }
    */
    
    // PLACEHOLDER - REPLACE WITH YOUR IMPLEMENTATION
    int query_pos = query_block_start * BLOCK_SIZE + threadIdx.x;
    if (batch_id < batch_size && query_pos < seq_len) {
        // Simple placeholder implementation
        for (int d = 0; d < head_dim; d++) {
            output[batch_id * seq_len * head_dim + query_pos * head_dim + d] = 
                Q[batch_id * seq_len * head_dim + query_pos * head_dim + d];
        }
    }
}

// Kernel 4: Student Exercise - Implement fused attention with dropout
__global__ void studentFusedAttentionDropout(float* Q, float* K, float* V, float* output, 
                                            float* dropout_mask, float dropout_prob,
                                            int batch_size, int seq_len, int head_dim) {
    // TODO: Implement fused attention with dropout in a single kernel
    // HINT: Generate random numbers and apply dropout mask during attention computation
    
    int batch_id = blockIdx.x;
    int query_pos = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_id < batch_size && query_pos < seq_len) {
        // FIX: Implement attention with integrated dropout
        // 1. Compute attention weights
        // 2. Generate random numbers for dropout
        // 3. Apply dropout mask to attention weights
        // 4. Compute final output
        
        // Placeholder implementation
        for (int d = 0; d < head_dim; d++) {
            output[batch_id * seq_len * head_dim + query_pos * head_dim + d] = 
                Q[batch_id * seq_len * head_dim + query_pos * head_dim + d];
        }
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int size, float start_val = 0.0f) {
    for (int i = 0; i < size; i++) {
        mat[i] = start_val + (i % 100) * 0.01f - 0.5f;  // Small random-like values centered at 0
    }
}

int main() {
    printf("=== Attention Mechanism Exercise ===\n");
    printf("Learn to implement efficient attention mechanisms.\n\n");

    // Setup parameters
    const int BATCH_SIZE = 2;
    const int SEQ_LEN = 128;
    const int HEAD_DIM = 64;
    const int QKV_SIZE = BATCH_SIZE * SEQ_LEN * HEAD_DIM;
    
    size_t bytes = QKV_SIZE * sizeof(float);
    size_t output_bytes = QKV_SIZE * sizeof(float);
    size_t mask_bytes = BATCH_SIZE * SEQ_LEN * SEQ_LEN * sizeof(float);
    
    // Allocate host memory
    float *h_Q, *h_K, *h_V, *h_output_naive, *h_output_fused, *h_output_flash, *h_output_dropout;
    float *h_dropout_mask;
    
    h_Q = (float*)malloc(bytes);
    h_K = (float*)malloc(bytes);
    h_V = (float*)malloc(bytes);
    h_output_naive = (float*)malloc(output_bytes);
    h_output_fused = (float*)malloc(output_bytes);
    h_output_flash = (float*)malloc(output_bytes);
    h_output_dropout = (float*)malloc(output_bytes);
    h_dropout_mask = (float*)malloc(mask_bytes);
    
    // Initialize matrices
    initMatrix(h_Q, QKV_SIZE, 0.1f);
    initMatrix(h_K, QKV_SIZE, 0.2f);
    initMatrix(h_V, QKV_SIZE, 0.3f);
    initMatrix(h_dropout_mask, BATCH_SIZE * SEQ_LEN * SEQ_LEN, 0.5f);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output_naive, *d_output_fused, *d_output_flash, *d_output_dropout;
    float *d_dropout_mask;
    
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_output_naive, output_bytes);
    cudaMalloc(&d_output_fused, output_bytes);
    cudaMalloc(&d_output_flash, output_bytes);
    cudaMalloc(&d_output_dropout, output_bytes);
    cudaMalloc(&d_dropout_mask, mask_bytes);
    
    // Copy input to device
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_dropout_mask, h_dropout_mask, mask_bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(32, 4);  // 32x4 threads for naive implementation
    dim3 gridSize(BATCH_SIZE, 1, 1);
    int linearBlockSize = 256;
    int linearGridSize = (SEQ_LEN + linearBlockSize - 1) / linearBlockSize;
    dim3 fusedGridSize(BATCH_SIZE, (SEQ_LEN + linearBlockSize - 1) / linearBlockSize);
    
    // Run naive attention kernel
    printf("Running naive attention kernel...\n");
    naiveAttention<<<gridSize, blockSize>>>(d_Q, d_K, d_V, d_output_naive, 
                                           BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Run fused attention kernel
    printf("Running fused attention kernel...\n");
    fusedAttention<<<fusedGridSize, linearBlockSize>>>(d_Q, d_K, d_V, d_output_fused,
                                                      BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student attention exercises (complete the code first!)...\n");
    
    // FlashAttention-style exercise
    dim3 flashGridSize(BATCH_SIZE, (SEQ_LEN + 64 - 1) / 64, 1);
    studentFlashAttention<<<flashGridSize, linearBlockSize>>>(d_Q, d_K, d_V, d_output_flash,
                                                             BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Fused attention with dropout exercise
    studentFusedAttentionDropout<<<fusedGridSize, linearBlockSize>>>(d_Q, d_K, d_V, d_output_dropout,
                                                                    d_dropout_mask, 0.1f,
                                                                    BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the attention implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_naive, d_output_naive, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused, d_output_fused, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_flash, d_output_flash, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_dropout, d_output_dropout, output_bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements of first sequence):\n");
    printf("Q values:   %.3f %.3f %.3f %.3f %.3f\n", 
           h_Q[0], h_Q[1], h_Q[2], h_Q[3], h_Q[4]);
    printf("Fused att:  %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_fused[0], h_output_fused[1], h_output_fused[2], h_output_fused[3], h_output_fused[4]);
    
    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_output_naive); 
    free(h_output_fused); free(h_output_flash); free(h_output_dropout);
    free(h_dropout_mask);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output_naive);
    cudaFree(d_output_fused); cudaFree(d_output_flash); cudaFree(d_output_dropout);
    cudaFree(d_dropout_mask);
    
    printf("\nExercise completed! Notice how fused implementations improve attention efficiency.\n");
    
    return 0;
}