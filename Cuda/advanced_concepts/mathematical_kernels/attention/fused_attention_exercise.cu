/*
 * Fused Attention (FlashAttention-Style) Exercise
 *
 * This exercise demonstrates how to implement efficient attention mechanisms
 * using techniques inspired by FlashAttention to reduce memory usage.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Kernel 1: Standard Attention (Materializes Full Attention Matrix)
__global__ void standardAttention(float* Q, float* K, float* V, float* output,
                                int batch_size, int seq_len, int head_dim) {
    int batch_id = blockIdx.x;
    int query_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_id < batch_size && query_idx < seq_len) {
        // Compute query vector
        float q_vec[64];  // Assuming max head_dim of 64
        for (int d = 0; d < head_dim; d++) {
            q_vec[d] = Q[batch_id * seq_len * head_dim + query_idx * head_dim + d];
        }
        
        // Compute attention weights
        float weights[512];  // Assuming max seq_len of 512
        float max_weight = -INFINITY;
        
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
        float weight_sum = 0.0f;
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
            output[batch_id * seq_len * head_dim + query_idx * head_dim + d] = out_val;
        }
    }
}

// Kernel 2: Block-wise Attention (Reduced Memory Footprint)
__global__ void blockwiseAttention(float* Q, float* K, float* V, float* output,
                                 int batch_size, int seq_len, int head_dim) {
    int batch_id = blockIdx.x;
    int query_block_start = blockIdx.y;
    
    const int BLOCK_SIZE = 64;  // Process 64 tokens at a time
    int query_start = query_block_start * BLOCK_SIZE;
    int query_end = min(query_start + BLOCK_SIZE, seq_len);
    
    if (batch_id < batch_size) {
        // Process in blocks to reduce memory usage
        for (int q_idx = query_start + threadIdx.x; q_idx < query_end; q_idx += blockDim.x) {
            // Load query vector
            float q_vec[64];
            for (int d = 0; d < head_dim; d++) {
                q_vec[d] = Q[batch_id * seq_len * head_dim + q_idx * head_dim + d];
            }
            
            // Initialize output vector
            float out_vec[64] = {0};
            float max_score = -INFINITY;
            float sum_exp_scores = 0.0f;
            
            // Process key-value pairs in blocks
            for (int kv_block_start = 0; kv_block_start < seq_len; kv_block_start += BLOCK_SIZE) {
                int kv_start = kv_block_start;
                int kv_end = min(kv_start + BLOCK_SIZE, seq_len);
                
                // Process this key-value block
                for (int k_idx = kv_start; k_idx < kv_end; k_idx++) {
                    // Compute attention score
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += q_vec[d] * 
                                 K[batch_id * seq_len * head_dim + k_idx * head_dim + d];
                    }
                    score = score / sqrtf((float)head_dim);
                    
                    // Update max for numerical stability
                    float old_max = max_score;
                    max_score = fmaxf(max_score, score);
                    
                    // Rescale previous values to account for new max
                    float rescale = expf(old_max - max_score);
                    sum_exp_scores *= rescale;
                    
                    // Add contribution of current key-value pair
                    float exp_score = expf(score - max_score);
                    sum_exp_scores += exp_score;
                    
                    for (int d = 0; d < head_dim; d++) {
                        out_vec[d] = out_vec[d] * rescale + 
                                    exp_score * V[batch_id * seq_len * head_dim + k_idx * head_dim + d];
                    }
                }
            }
            
            // Normalize output
            for (int d = 0; d < head_dim; d++) {
                output[batch_id * seq_len * head_dim + q_idx * head_dim + d] = 
                    out_vec[d] / sum_exp_scores;
            }
        }
    }
}

// Kernel 3: Student Exercise - Implement FlashAttention-style algorithm with shared memory
__global__ void studentFlashAttention(float* Q, float* K, float* V, float* output,
                                    int batch_size, int seq_len, int head_dim) {
    // TODO: Implement a FlashAttention-style algorithm using shared memory tiling
    // HINT: Process attention computation in blocks to reduce memory usage
    
    int batch_id = blockIdx.x;
    int query_block_start = blockIdx.y;
    int key_value_block_start = blockIdx.z;
    
    const int BLOCK_SIZE = 64;  // Process 64 tokens at a time
    extern __shared__ float shared_mem[];
    
    // FIX: Partition shared memory for Q, K, V blocks and intermediate results
    // Calculate how much shared memory we need based on BLOCK_SIZE and head_dim
    int shared_offset = 0;
    float* Q_block = &shared_mem[shared_offset];
    shared_offset += BLOCK_SIZE * head_dim;
    
    float* K_block = &shared_mem[shared_offset];
    shared_offset += BLOCK_SIZE * head_dim;
    
    float* V_block = &shared_mem[shared_offset];
    shared_offset += BLOCK_SIZE * head_dim;
    
    float* O_block = &shared_mem[shared_offset];
    shared_offset += BLOCK_SIZE * head_dim;
    
    float* L_block = &shared_mem[shared_offset];  // Normalization terms
    shared_offset += BLOCK_SIZE;
    
    float* M_block = &shared_mem[shared_offset];  // Max values
    
    int query_start = query_block_start * BLOCK_SIZE;
    int key_start = key_value_block_start * BLOCK_SIZE;
    
    if (batch_id < batch_size && query_start < seq_len && key_start < seq_len) {
        // FIX: Load block of Q vectors into shared memory
        for (int i = threadIdx.x; i < min(BLOCK_SIZE, seq_len - query_start); i += blockDim.x) {
            int q_idx = query_start + i;
            if (q_idx < seq_len) {
                for (int d = 0; d < head_dim; d++) {
                    Q_block[i * head_dim + d] = 
                        Q[batch_id * seq_len * head_dim + q_idx * head_dim + d];
                }
            }
        }
        
        // FIX: Load block of K and V vectors into shared memory
        for (int i = threadIdx.x; i < min(BLOCK_SIZE, seq_len - key_start); i += blockDim.x) {
            int k_idx = key_start + i;
            if (k_idx < seq_len) {
                for (int d = 0; d < head_dim; d++) {
                    K_block[i * head_dim + d] = 
                        K[batch_id * seq_len * head_dim + k_idx * head_dim + d];
                    V_block[i * head_dim + d] = 
                        V[batch_id * seq_len * head_dim + k_idx * head_dim + d];
                }
            }
        }
        
        __syncthreads();
        
        // FIX: Initialize output and normalization variables
        for (int i = threadIdx.x; i < min(BLOCK_SIZE, seq_len - query_start); i += blockDim.x) {
            int q_idx = query_start + i;
            if (q_idx < seq_len) {
                // Initialize output for this query
                for (int d = 0; d < head_dim; d++) {
                    O_block[i * head_dim + d] = 0.0f;
                }
                L_block[i] = 0.0f;  // Normalization term
                M_block[i] = -INFINITY;  // Maximum value
            }
        }
        
        __syncthreads();
        
        // FIX: Compute attention scores and update output in an online fashion
        for (int i = threadIdx.x; i < min(BLOCK_SIZE, seq_len - query_start); i += blockDim.x) {
            int q_idx = query_start + i;
            if (q_idx < seq_len) {
                for (int j = 0; j < min(BLOCK_SIZE, seq_len - key_start); j++) {
                    int k_idx = key_start + j;
                    if (k_idx < seq_len) {
                        // Compute attention score
                        float score = 0.0f;
                        for (int d = 0; d < head_dim; d++) {
                            score += Q_block[i * head_dim + d] * K_block[j * head_dim + d];
                        }
                        score = score / sqrtf((float)head_dim);
                        
                        // Update normalization constants
                        float old_max = M_block[i];
                        float old_norm = L_block[i];
                        float new_max = fmaxf(M_block[i], score);
                        
                        // Rescale previous values
                        float rescale = expf(old_max - new_max);
                        float new_norm = old_norm * rescale;
                        
                        // Add contribution of current key-value pair
                        float exp_score = expf(score - new_max);
                        new_norm += exp_score;
                        
                        // Update output vector
                        for (int d = 0; d < head_dim; d++) {
                            O_block[i * head_dim + d] = 
                                O_block[i * head_dim + d] * rescale + 
                                exp_score * V_block[j * head_dim + d];
                        }
                        
                        // Update normalization constants
                        L_block[i] = new_norm;
                        M_block[i] = new_max;
                    }
                }
            }
        }
        
        __syncthreads();
        
        // FIX: Write results back to global memory
        for (int i = threadIdx.x; i < min(BLOCK_SIZE, seq_len - query_start); i += blockDim.x) {
            int q_idx = query_start + i;
            if (q_idx < seq_len) {
                // Normalize and store output
                for (int d = 0; d < head_dim; d++) {
                    output[batch_id * seq_len * head_dim + q_idx * head_dim + d] = 
                        O_block[i * head_dim + d] / L_block[i];
                }
            }
        }
    }
}

// Kernel 4: Student Exercise - Implement causal (masked) attention
__global__ void studentCausalAttention(float* Q, float* K, float* V, float* output,
                                     int batch_size, int seq_len, int head_dim) {
    // TODO: Implement causal attention that only attends to previous positions
    // HINT: Apply a causal mask during attention computation
    
    int batch_id = blockIdx.x;
    int query_idx = blockIdx.y * blockDim.x + threadIdx.x;
    
    if (batch_id < batch_size && query_idx < seq_len) {
        // FIX: Load query vector
        float q_vec[64];
        for (int d = 0; d < head_dim; d++) {
            q_vec[d] = Q[batch_id * seq_len * head_dim + query_idx * head_dim + d];
        }
        
        // FIX: Initialize output vector and normalization variables
        float out_vec[64] = {0};
        float max_score = -INFINITY;
        float sum_exp_scores = 0.0f;
        
        // FIX: Only attend to positions <= query_idx (causal mask)
        for (int k_idx = 0; k_idx <= query_idx; k_idx++) {
            // Compute attention score
            float score = 0.0f;
            for (int d = 0; d < head_dim; d++) {
                score += q_vec[d] * 
                         K[batch_id * seq_len * head_dim + k_idx * head_dim + d];
            }
            score = score / sqrtf((float)head_dim);
            
            // Apply causal mask: negative infinity for future positions
            // (already handled by the loop condition: k_idx <= query_idx)
            
            // Update max for numerical stability
            float old_max = max_score;
            max_score = fmaxf(max_score, score);
            
            // Rescale previous values to account for new max
            float rescale = expf(old_max - max_score);
            sum_exp_scores *= rescale;
            
            // Add contribution of current key-value pair
            float exp_score = expf(score - max_score);
            sum_exp_scores += exp_score;
            
            for (int d = 0; d < head_dim; d++) {
                out_vec[d] = out_vec[d] * rescale + 
                            exp_score * V[batch_id * seq_len * head_dim + k_idx * head_dim + d];
            }
        }
        
        // FIX: Normalize output
        for (int d = 0; d < head_dim; d++) {
            output[batch_id * seq_len * head_dim + query_idx * head_dim + d] = 
                out_vec[d] / sum_exp_scores;
        }
    }
}

// Utility function to initialize matrix
void initMatrix(float* mat, int rows, int cols, float start_val = 0.0f) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = start_val + (i + j) * 0.01f - 0.5f;
        }
    }
}

int main() {
    printf("=== Fused Attention (FlashAttention-Style) Exercise ===\n");
    printf("Learn to implement efficient attention mechanisms.\n\n");

    // Setup parameters
    const int BATCH_SIZE = 1;
    const int SEQ_LEN = 128;
    const int HEAD_DIM = 64;
    const int QKV_SIZE = BATCH_SIZE * SEQ_LEN * HEAD_DIM;
    
    size_t bytes = QKV_SIZE * sizeof(float);
    size_t output_bytes = QKV_SIZE * sizeof(float);
    
    // Allocate host memory
    float *h_Q, *h_K, *h_V, *h_output_standard, *h_output_blockwise, *h_output_flash, *h_output_causal;
    h_Q = (float*)malloc(bytes);
    h_K = (float*)malloc(bytes);
    h_V = (float*)malloc(bytes);
    h_output_standard = (float*)malloc(output_bytes);
    h_output_blockwise = (float*)malloc(output_bytes);
    h_output_flash = (float*)malloc(output_bytes);
    h_output_causal = (float*)malloc(output_bytes);
    
    // Initialize matrices
    initMatrix(h_Q, BATCH_SIZE, SEQ_LEN * HEAD_DIM, 0.1f);
    initMatrix(h_K, BATCH_SIZE, SEQ_LEN * HEAD_DIM, 0.2f);
    initMatrix(h_V, BATCH_SIZE, SEQ_LEN * HEAD_DIM, 0.3f);
    
    // Initialize output matrices to zero
    memset(h_output_standard, 0, output_bytes);
    memset(h_output_blockwise, 0, output_bytes);
    memset(h_output_flash, 0, output_bytes);
    memset(h_output_causal, 0, output_bytes);
    
    // Allocate device memory
    float *d_Q, *d_K, *d_V, *d_output_standard, *d_output_blockwise, *d_output_flash, *d_output_causal;
    cudaMalloc(&d_Q, bytes);
    cudaMalloc(&d_K, bytes);
    cudaMalloc(&d_V, bytes);
    cudaMalloc(&d_output_standard, output_bytes);
    cudaMalloc(&d_output_blockwise, output_bytes);
    cudaMalloc(&d_output_flash, output_bytes);
    cudaMalloc(&d_output_causal, output_bytes);
    
    // Copy matrices to device
    cudaMemcpy(d_Q, h_Q, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_K, h_K, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_V, h_V, bytes, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    int blockSize = 256;
    int gridSize = (SEQ_LEN + blockSize - 1) / blockSize;
    dim3 attnGridSize(BATCH_SIZE, (SEQ_LEN + 63) / 64, (SEQ_LEN + 63) / 64);  // For FlashAttention
    
    // Run standard attention kernel
    printf("Running standard attention kernel...\n");
    standardAttention<<<dim3(BATCH_SIZE, gridSize), blockSize>>>(d_Q, d_K, d_V, d_output_standard,
                                                               BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Run block-wise attention kernel
    printf("Running block-wise attention kernel...\n");
    blockwiseAttention<<<dim3(BATCH_SIZE, gridSize), blockSize>>>(d_Q, d_K, d_V, d_output_blockwise,
                                                                 BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student attention exercises (complete the code first!)...\n");
    
    // FlashAttention-style exercise
    size_t shared_mem_size = (2 * 64 * HEAD_DIM + 64 * HEAD_DIM + 64 * HEAD_DIM + 64 + 64) * sizeof(float);
    studentFlashAttention<<<attnGridSize, blockSize, shared_mem_size>>>(d_Q, d_K, d_V, d_output_flash,
                                                                      BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Causal attention exercise
    studentCausalAttention<<<dim3(BATCH_SIZE, gridSize), blockSize>>>(d_Q, d_K, d_V, d_output_causal,
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
    cudaMemcpy(h_output_standard, d_output_standard, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_blockwise, d_output_blockwise, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_flash, d_output_flash, output_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_causal, d_output_causal, output_bytes, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements of first sequence):\n");
    printf("Standard: %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_standard[0], h_output_standard[1], h_output_standard[2], h_output_standard[3], h_output_standard[4]);
    printf("Blockwise: %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_blockwise[0], h_output_blockwise[1], h_output_blockwise[2], h_output_blockwise[3], h_output_blockwise[4]);
    printf("FlashAtt: %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_flash[0], h_output_flash[1], h_output_flash[2], h_output_flash[3], h_output_flash[4]);
    
    // Cleanup
    free(h_Q); free(h_K); free(h_V); free(h_output_standard); 
    free(h_output_blockwise); free(h_output_flash); free(h_output_causal);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_output_standard);
    cudaFree(d_output_blockwise); cudaFree(d_output_flash); cudaFree(d_output_causal);
    
    printf("\nExercise completed! Notice how FlashAttention reduces memory usage.\n");
    
    return 0;
}