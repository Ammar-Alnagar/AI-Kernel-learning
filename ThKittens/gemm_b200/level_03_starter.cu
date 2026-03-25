/*
 * ThunderKittens Fill-in Tutorial (B200) - Level 03
 * Goal: Add shared-memory tiling
 *
 * How to use:
 * 1) Open this starter file and replace TODO markers with working code.
 * 2) If blocked, compare against level_03_solution.cu in the same folder.
 * 3) Build with: make LEVEL=03 TRACK=starter clean && make LEVEL=03 TRACK=starter run
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
    __shared__ __nv_bfloat16 As[BLOCK_SIZE][BLOCK_SIZE], Bs[BLOCK_SIZE][BLOCK_SIZE];
    int tx = threadIdx.x, bx = blockIdx.x;
    int by = blockIdx.y, ty = threadIdx.y;
    int row = by * BLOCK_SIZE + ty;
    int col = bx * BLOCK_SIZE + tx;

    float sum = 0.0f;
    for (int tile = 0; tile < (N + BLOCK_SIZE - 1) / BLOCK_SIZE; ++tile) {
        As[ty][tx] = A[row * N + tile * BLOCK_SIZE + tx];
        Bs[ty][tx] = B[(tile * BLOCK_SIZE + ty) * N + col];
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            sum += __bfloat162float(As[ty][k] * Bs[k][tx]);
        }
        __syncthreads();
    }
    C[row * N + col] = __float2bfloat16(sum);
}

void matmul(__nv_bfloat16* A, __nv_bfloat16* B, __nv_bfloat16* C, int N) {
    dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    kernel<<<blocks, threads>>>(A, B, C, N);
}

#include "launch.cu"
