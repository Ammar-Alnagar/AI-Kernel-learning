/*
 * ThunderKittens Fill-in Tutorial (B200) - Level 02
 * Goal: Simple for-loop GEMM (bf16) baseline
 *
 * How to use:
 * 1) Open this starter file and replace TODO markers with working code.
 * 2) If blocked, compare against level_02_solution.cu in the same folder.
 * 3) Build with: make LEVEL=02 TRACK=starter clean && make LEVEL=02 TRACK=starter run
 *
 * Exercise rules:
 * - Keep tensor/semaphore phase ordering correct.
 * - Keep destination-first TK op signatures.
 * - Do not change launch harness in launch.cu.
 */

#include <cuda_runtime.h>
#include <cuda_bf16.h>

static constexpr int BLOCK_SIZE = 32;

__global__ void kernel(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < N && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < N; k++) {
            sum += __bfloat162float(A[row * N + k] * B[k * N + col]);
        }
        C[row * N + col] = __float2bfloat16(sum);
    }
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + (BLOCK_SIZE-1)) / BLOCK_SIZE, (N + (BLOCK_SIZE-1)) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "launch.cu"
