/*
 * Fused Matmul + Softmax Exercise
 *
 * This exercise demonstrates how to fuse matrix multiplication and softmax operations
 * to reduce memory traffic and improve performance.
 */

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// Kernel 1: Unfused Matmul + Softmax (Two Separate Kernels)
__global__ void matmulKernel(float* A, float* B, float* temp, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        temp[row * N + col] = sum;
    }
}

__global__ void softmaxKernel(float* input, float* output, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < rows) {
        // Find max for numerical stability
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
        
        // Apply softmax
        for (int j = 0; j < cols; j++) {
            output[row * cols + j] = expf(input[row * cols + j] - max_val) / sum;
        }
    }
}

// Kernel 2: Fused Matmul + Softmax (Single Kernel)
__global__ void fusedMatmulSoftmax(float* A, float* B, float* output, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // Compute matmul result
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            sum += A[row * K + k] * B[k * N + col];
        }
        
        // Store temporarily in shared memory for softmax computation
        __shared__ float row_scores[1024];  // Assuming max 1024 cols per row
        if (threadIdx.y == 0) {  // Only one thread per row stores the full row
            for (int c = 0; c < N; c++) {
                float temp_sum = 0.0f;
                for (int k = 0; k < K; k++) {
                    temp_sum += A[row * K + k] * B[k * N + c];
                }
                row_scores[c] = temp_sum;
            }
        }
        __syncthreads();
        
        // Find max for numerical stability
        float max_val = row_scores[col];
        if (threadIdx.y == 0) {  // Only one thread finds max
            for (int c = 0; c < N; c++) {
                if (row_scores[c] > max_val) {
                    max_val = row_scores[c];
                }
            }
        }
        __syncthreads();
        
        // Compute sum of exponentials
        float sum = 0.0f;
        if (threadIdx.y == 0) {  // Only one thread computes sum
            for (int c = 0; c < N; c++) {
                sum += expf(row_scores[c] - max_val);
            }
        }
        __syncthreads();
        
        // Apply softmax
        output[row * N + col] = expf(row_scores[col] - max_val) / sum;
    }
}

// Kernel 3: Improved Fused Matmul + Softmax (Better Memory Access)
__global__ void improvedFusedMatmulSoftmax(float* A, float* B, float* output, int M, int N, int K) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M) {
        // Compute entire row of matmul result
        float* row_buffer = new float[N];  // Dynamic allocation in kernel (not recommended in practice)
        
        for (int col = 0; col < N; col++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[row * K + k] * B[k * N + col];
            }
            row_buffer[col] = sum;
        }
        
        // Find max for numerical stability
        float max_val = row_buffer[0];
        for (int j = 1; j < N; j++) {
            if (row_buffer[j] > max_val) {
                max_val = row_buffer[j];
            }
        }
        
        // Compute sum of exponentials
        float sum = 0.0f;
        for (int j = 0; j < N; j++) {
            sum += expf(row_buffer[j] - max_val);
        }
        
        // Apply softmax
        for (int j = 0; j < N; j++) {
            output[row * N + j] = expf(row_buffer[j] - max_val) / sum;
        }
        
        delete[] row_buffer;
    }
}

// Kernel 4: Student Exercise - Implement efficient fused matmul + softmax with shared memory
__global__ void studentFusedMatmulSoftmax(float* A, float* B, float* output, int M, int N, int K) {
    // TODO: Implement an efficient fused matmul + softmax using shared memory tiling
    // HINT: Use shared memory to cache portions of A and B matrices
    
    #define TILE_SIZE 16
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (row < M && col < N) {
        // FIX: Implement tiled matmul computation
        float sum = 0.0f;
        
        // Loop over tiles
        for (int t = 0; t < (K + TILE_SIZE - 1) / TILE_SIZE; t++) {
            // Load tiles into shared memory
            if (row < M && t * TILE_SIZE + threadIdx.x < K) {
                As[threadIdx.y][threadIdx.x] = A[row * K + t * TILE_SIZE + threadIdx.x];
            } else {
                As[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            if (t * TILE_SIZE + threadIdx.y < K && col < N) {
                Bs[threadIdx.y][threadIdx.x] = B[(t * TILE_SIZE + threadIdx.y) * N + col];
            } else {
                Bs[threadIdx.y][threadIdx.x] = 0.0f;
            }
            
            __syncthreads();
            
            // Compute partial result for this tile
            for (int k = 0; k < TILE_SIZE; k++) {
                sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
            }
            
            __syncthreads();
        }
        
        // FIX: Instead of storing to output directly, collect all values in shared memory
        // for the softmax computation to avoid redundant matmul computation
        // This is challenging - consider how to organize threads to compute softmax
        // for the entire row after matmul is complete
        
        // For now, just store the matmul result
        // In a complete implementation, you'd need a more sophisticated approach
        // to compute softmax on the entire row of results
        output[row * N + col] = sum;
    }
    #undef TILE_SIZE
}

// Kernel 5: Student Exercise - Implement FlashAttention-style fused operation
__global__ void studentFlashAttentionStyle(float* Q, float* K, float* V, float* output, 
                                        int batch_size, int seq_len, int head_dim) {
    // TODO: Implement a simplified FlashAttention-style fused operation
    // HINT: Process attention computation in blocks to avoid materializing full attention matrix
    
    int batch_id = blockIdx.x;
    int query_block_start = blockIdx.y;
    int key_block_start = blockIdx.z;
    
    // FIX: Implement block-wise attention computation
    // Process queries and keys in blocks to reduce memory footprint
    // Use online softmax computation to avoid materializing full attention matrix
    
    // Define block size for processing
    const int BLOCK_SIZE = 64;  // Process 64 tokens at a time
    
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
    
    // Process attention in blocks
    for (int q_start = query_block_start * BLOCK_SIZE; 
         q_start < min((query_block_start + 1) * BLOCK_SIZE, seq_len); 
         q_start += BLOCK_SIZE) {
        
        // Load query block
        for (int i = 0; i < BLOCK_SIZE && q_start + i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                Q_block[i * head_dim + d] = 
                    Q[batch_id * seq_len * head_dim + (q_start + i) * head_dim + d];
            }
        }
        
        // Initialize output for this block
        for (int i = 0; i < BLOCK_SIZE && q_start + i < seq_len; i++) {
            for (int d = 0; d < head_dim; d++) {
                O_block[i * head_dim + d] = 0.0f;
                L_block[i] = 0.0f;  // Normalization term
                M_block[i] = -INFINITY;  // Maximum value
            }
        }
        
        // Iterate through key/value blocks
        for (int kv_start = 0; kv_start < seq_len; kv_start += BLOCK_SIZE) {
            // Load key and value blocks
            for (int i = 0; i < BLOCK_SIZE && kv_start + i < seq_len; i++) {
                for (int d = 0; d < head_dim; d++) {
                    K_block[i * head_dim + d] = 
                        K[batch_id * seq_len * head_dim + (kv_start + i) * head_dim + d];
                    V_block[i * head_dim + d] = 
                        V[batch_id * seq_len * head_dim + (kv_start + i) * head_dim + d];
                }
            }
            
            // Compute attention scores and update output in an online fashion
            for (int qi = 0; qi < BLOCK_SIZE && q_start + qi < seq_len; qi++) {
                float old_max = M_block[qi];
                float old_norm = L_block[qi];
                float new_max = old_max;
                
                for (int ki = 0; ki < BLOCK_SIZE && kv_start + ki < seq_len; ki++) {
                    // Compute attention score
                    float score = 0.0f;
                    for (int d = 0; d < head_dim; d++) {
                        score += Q_block[qi * head_dim + d] * K_block[ki * head_dim + d];
                    }
                    score = score / sqrtf(head_dim);
                    
                    new_max = fmaxf(new_max, score);
                }
                
                // Update normalization constants
                float rescale = expf(old_max - new_max);
                float norm_new = 0.0f;
                
                // Update output values
                for (int d = 0; d < head_dim; d++) {
                    float out_old = O_block[qi * head_dim + d] * rescale;
                    float out_new = 0.0f;
                    
                    for (int ki = 0; ki < BLOCK_SIZE && kv_start + ki < seq_len; ki++) {
                        // Compute attention score
                        float score = 0.0f;
                        for (int d2 = 0; d2 < head_dim; d2++) {
                            score += Q_block[qi * head_dim + d2] * K_block[ki * head_dim + d2];
                        }
                        score = score / sqrtf(head_dim);
                        
                        float weight = expf(score - new_max);
                        norm_new += weight;
                        out_new += weight * V_block[ki * head_dim + d];
                    }
                    
                    O_block[qi * head_dim + d] = out_old + out_new;
                }
                
                L_block[qi] = old_norm * rescale + norm_new;
                M_block[qi] = new_max;
            }
        }
        
        // Normalize and store final output
        for (int qi = 0; qi < BLOCK_SIZE && q_start + qi < seq_len; qi++) {
            for (int d = 0; d < head_dim; d++) {
                output[batch_id * seq_len * head_dim + (q_start + qi) * head_dim + d] = 
                    O_block[qi * head_dim + d] / L_block[qi];
            }
        }
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

// Utility function to initialize matrix
void initMatrix(float* mat, int rows, int cols, float start_val = 0.0f) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i * cols + j] = start_val + (i + j) * 0.01f - 0.5f;
        }
    }
}

int main() {
    printf("=== Fused Matmul + Softmax Exercise ===\n");
    printf("Learn to implement fused operations for improved performance.\n\n");

    // Setup parameters
    const int M = 128, N = 128, K = 128;  // Matrix dimensions
    size_t bytes_A = M * K * sizeof(float);
    size_t bytes_B = K * N * sizeof(float);
    size_t bytes_C = M * N * sizeof(float);
    size_t bytes_temp = M * N * sizeof(float);
    
    // Allocate host memory
    float *h_A, *h_B, *h_temp, *h_output_unfused, *h_output_fused, *h_output_improved, *h_output_student;
    h_A = (float*)malloc(bytes_A);
    h_B = (float*)malloc(bytes_B);
    h_temp = (float*)malloc(bytes_temp);
    h_output_unfused = (float*)malloc(bytes_C);
    h_output_fused = (float*)malloc(bytes_C);
    h_output_improved = (float*)malloc(bytes_C);
    h_output_student = (float*)malloc(bytes_C);
    
    // Initialize matrices
    initMatrix(h_A, M, K, 0.1f);
    initMatrix(h_B, K, N, 0.2f);
    
    // Initialize output matrices to zero
    memset(h_output_unfused, 0, bytes_C);
    memset(h_output_fused, 0, bytes_C);
    memset(h_output_improved, 0, bytes_C);
    memset(h_output_student, 0, bytes_C);
    
    // Allocate device memory
    float *d_A, *d_B, *d_temp, *d_output_unfused, *d_output_fused, *d_output_improved, *d_output_student;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_B);
    cudaMalloc(&d_temp, bytes_temp);
    cudaMalloc(&d_output_unfused, bytes_C);
    cudaMalloc(&d_output_fused, bytes_C);
    cudaMalloc(&d_output_improved, bytes_C);
    cudaMalloc(&d_output_student, bytes_C);
    
    // Copy matrices to device
    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_B, cudaMemcpyHostToDevice);
    
    // Define block and grid dimensions
    dim3 blockSize(16, 16);  // 16x16 threads per block
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, 
                  (M + blockSize.y - 1) / blockSize.y);
    
    int linearBlockSize = 256;
    int linearGridSize = (M + linearBlockSize - 1) / linearBlockSize;
    
    // Run unfused matmul + softmax (two kernels)
    printf("Running unfused matmul + softmax kernels...\n");
    matmulKernel<<<gridSize, blockSize>>>(d_A, d_B, d_temp, M, N, K);
    cudaDeviceSynchronize();
    softmaxKernel<<<linearGridSize, linearBlockSize>>>(d_temp, d_output_unfused, M, N);
    cudaDeviceSynchronize();
    
    // Run fused matmul + softmax kernel
    printf("Running fused matmul + softmax kernel...\n");
    fusedMatmulSoftmax<<<gridSize, blockSize>>>(d_A, d_B, d_output_fused, M, N, K);
    cudaDeviceSynchronize();
    
    // Run improved fused matmul + softmax kernel
    printf("Running improved fused matmul + softmax kernel...\n");
    improvedFusedMatmulSoftmax<<<linearGridSize, linearBlockSize>>>(d_A, d_B, d_output_improved, M, N, K);
    cudaDeviceSynchronize();
    
    // Run student exercises (will fail to compile until completed)
    printf("Running student fused operation exercises (complete the code first!)...\n");
    
    // Fused matmul + softmax exercise
    studentFusedMatmulSoftmax<<<gridSize, blockSize>>>(d_A, d_B, d_output_student, M, N, K);
    cudaDeviceSynchronize();
    
    // FlashAttention-style exercise
    const int BATCH_SIZE = 1;
    const int SEQ_LEN = 128;
    const int HEAD_DIM = 64;
    size_t qkv_bytes = BATCH_SIZE * SEQ_LEN * HEAD_DIM * sizeof(float);
    float *d_Q, *d_K, *d_V, *d_attn_output;
    cudaMalloc(&d_Q, qkv_bytes);
    cudaMalloc(&d_K, qkv_bytes);
    cudaMalloc(&d_V, qkv_bytes);
    cudaMalloc(&d_attn_output, qkv_bytes);
    
    initMatrix((float*)d_Q, BATCH_SIZE * SEQ_LEN, HEAD_DIM, 0.1f);
    initMatrix((float*)d_K, BATCH_SIZE * SEQ_LEN, HEAD_DIM, 0.2f);
    initMatrix((float*)d_V, BATCH_SIZE * SEQ_LEN, HEAD_DIM, 0.3f);
    
    dim3 attn_gridSize(BATCH_SIZE, (SEQ_LEN + 64 - 1) / 64, (SEQ_LEN + 64 - 1) / 64);
    studentFlashAttentionStyle<<<attn_gridSize, linearBlockSize>>>(d_Q, d_K, d_V, d_attn_output, 
                                                                 BATCH_SIZE, SEQ_LEN, HEAD_DIM);
    cudaDeviceSynchronize();
    
    // Check for kernel launch errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Student exercise kernel execution failed: %s\n", cudaGetErrorString(err));
        printf("Hint: Complete the fused operation implementations in the student exercises!\n");
    } else {
        printf("Student exercise kernels executed successfully!\n");
    }
    
    // Copy results back to host
    cudaMemcpy(h_output_unfused, d_output_unfused, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_fused, d_output_fused, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_improved, d_output_improved, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_output_student, d_output_student, bytes_C, cudaMemcpyDeviceToHost);
    
    // Print sample results
    printf("\nSample results (first 5 elements of first row):\n");
    printf("Unfused:   %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_unfused[0], h_output_unfused[1], h_output_unfused[2], h_output_unfused[3], h_output_unfused[4]);
    printf("Fused:     %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_fused[0], h_output_fused[1], h_output_fused[2], h_output_fused[3], h_output_fused[4]);
    printf("Improved:  %.3f %.3f %.3f %.3f %.3f\n", 
           h_output_improved[0], h_output_improved[1], h_output_improved[2], h_output_improved[3], h_output_improved[4]);
    
    // Cleanup
    free(h_A); free(h_B); free(h_temp); free(h_output_unfused); 
    free(h_output_fused); free(h_output_improved); free(h_output_student);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_temp); cudaFree(d_output_unfused);
    cudaFree(d_output_fused); cudaFree(d_output_improved); cudaFree(d_output_student);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_attn_output);
    
    printf("\nExercise completed! Notice how fusing operations can improve performance.\n");
    
    return 0;
}