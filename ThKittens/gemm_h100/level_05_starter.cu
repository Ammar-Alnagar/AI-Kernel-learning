/*
 * ThunderKittens Fill-in Tutorial (H100) - Level 05
 * Goal: Use Tensor Cores at warpgroup scope (warpgroup::mma_AB)
 *
 * How to use:
 * 1) Open this starter file and replace TODO markers with working code.
 * 2) If blocked, compare against level_05_solution.cu in the same folder.
 * 3) Build with: make LEVEL=05 TRACK=starter clean && make LEVEL=05 TRACK=starter run
 *
 * Exercise rules:
 * - Keep destination-first TK op signatures.
 * - Preserve synchronization and semaphore correctness.
 * - Do not change launch harness in launch.cu.
 */

#include <iostream>
#include <random>
#include <cuda_bf16.h>
#include <chrono>
#include <cuda_runtime.h>

#include "kittens.cuh"
using namespace kittens;

static constexpr int BLOCK_SIZE = 64;
static constexpr int NUM_WORKERS =  (4);
static constexpr int NUM_THREADS = (NUM_WORKERS*kittens::WARP_THREADS);

struct matmul_globals { 
    using sub_tile = st_bf<BLOCK_SIZE,BLOCK_SIZE>;
    using tile_gl =  gl<bf16,  1, 1, -1, -1, sub_tile>;
    tile_gl A;
    tile_gl B; 
    tile_gl C;
    int N;
};

__global__ void kernel(const __grid_constant__ matmul_globals g) {

    extern __shared__ alignment_dummy __shm[]; 
    shared_allocator al((int*)&__shm[0]);
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &As = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>(); 
    st_bf<BLOCK_SIZE,BLOCK_SIZE> &Bs = al.allocate<st_bf<BLOCK_SIZE,BLOCK_SIZE>>(); 
    
    rt_fl<16,BLOCK_SIZE> C_accum;
    rt_fl<16,BLOCK_SIZE> C_accum_cpy;

    int bx = blockIdx.x; 
    int by = blockIdx.y; 
    int row = by; 
    int col = bx; 

    // int condition = (threadIdx.x == 0 && threadIdx.y == 0 & blockIdx.x == 0);

    // TODO(L05-1): zero final accumulator before K loop.
    // kittens::warp::zero(C_accum_cpy);
    int num_tiles = (g.N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < num_tiles; ++tile) {
        warpgroup::load(As, g.A, {0, 0, row, tile});
        warpgroup::load(Bs, g.B, {0, 0, tile, col});
        __syncthreads();
        // TODO(L05-2): issue warpgroup MMA on shared tiles.
        // warpgroup::mma_AB(C_accum, As, Bs);
        // TODO(L05-3): wait for async warpgroup MMA completion.
        // warpgroup::mma_async_wait();
        kittens::warp::add(C_accum_cpy, C_accum_cpy, C_accum);
        kittens::warp::zero(C_accum);
    }
    warpgroup::store(g.C, C_accum_cpy, {0, 0, row, col});
}

// launch kernel
void matmul(bf16* A, bf16* B, bf16* C, size_t N) { 

    // global pointers
    using a_gl = matmul_globals::tile_gl;
    using b_gl = matmul_globals::tile_gl; 
    using c_gl = matmul_globals::tile_gl;
    a_gl  a_arg{A, nullptr, nullptr, N, N};
    b_gl  b_arg{B, nullptr, nullptr, N, N};
    c_gl  c_arg{C, nullptr, nullptr, N, N};
    matmul_globals g{a_arg, b_arg, c_arg, (int)N}; 

    // launch
    dim3 blocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    unsigned long mem_size = 100000;
    cudaDeviceSynchronize();
    cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, mem_size);
    kernel<<<blocks, NUM_THREADS, mem_size>>>(g);
    CHECK_CUDA_ERROR(cudaGetLastError());
    cudaDeviceSynchronize();
}

#include "launch.cu"
