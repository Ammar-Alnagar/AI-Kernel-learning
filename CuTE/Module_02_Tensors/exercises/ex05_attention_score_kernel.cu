#include <cmath>
#include <cstdio>
#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>
#include <vector>

using namespace cute;

constexpr int HEAD_DIM = 128;
constexpr int NUM_QUERIES = 1024;
constexpr int BLOCK_THREADS = 256;
constexpr int WARP_SIZE = 32;
constexpr int WARPS_PER_BLOCK = BLOCK_THREADS / WARP_SIZE; // 8
constexpr int ELEMS_PER_THREAD = HEAD_DIM / WARP_SIZE;     // 4

__global__ void attention_score_kernel(const float *__restrict__ Q_data,
                                       const float *__restrict__ K_data,
                                       float *__restrict__ scores,
                                       float scale) {
  auto Q =
      make_tensor(make_gmem_ptr(Q_data),
                  make_layout(make_shape(Int<NUM_QUERIES>{}, Int<HEAD_DIM>{}),
                              GenRowMajor{}));
  auto K = make_tensor(make_gmem_ptr(K_data),
                       make_layout(make_shape(Int<HEAD_DIM>{})));
  auto S = make_tensor(make_gmem_ptr(scores),
                       make_layout(make_shape(Int<NUM_QUERIES>{})));

  int tid = threadIdx.x;
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;

  int query_idx = blockIdx.x * WARPS_PER_BLOCK + warp_id;
  if (query_idx >= NUM_QUERIES)
    return;

  // local_tile: grab this warp's query row
  auto q_row = local_tile(Q, make_shape(Int<1>{}, Int<HEAD_DIM>{}),
                          make_coord(query_idx, 0));

  // local_partition: split head_dim across 32 warp lanes
  auto thr_layout_2d = make_layout(make_shape(Int<1>{}, Int<WARP_SIZE>{}));
  auto q_local = local_partition(q_row, thr_layout_2d, lane_id);

  auto thr_layout_1d = make_layout(make_shape(Int<WARP_SIZE>{}));
  auto k_local = local_partition(K, thr_layout_1d, lane_id);

  // Partial dot product: 4 MACs per thread
  float partial = 0.0f;
#pragma unroll
  for (int i = 0; i < ELEMS_PER_THREAD; i++) {
    partial += q_local(0, i) * k_local(i);
  }

// Warp reduction via register shuffle
#pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    partial += __shfl_down_sync(0xFFFFFFFF, partial, offset);
  }

  if (lane_id == 0) {
    S(query_idx) = partial * scale;
  }
}

void cpu_attention_scores(const std::vector<float> &Q,
                          const std::vector<float> &K,
                          std::vector<float> &scores, float scale) {
  for (int i = 0; i < NUM_QUERIES; i++) {
    float dot = 0.0f;
    for (int d = 0; d < HEAD_DIM; d++) {
      dot += Q[i * HEAD_DIM + d] * K[d];
    }
    scores[i] = dot * scale;
  }
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("=== Attention Score Kernel ===\n");
  printf("GPU: %s\n", prop.name);
  printf("SMs: %d | Peak BW: %.1f GB/s\n\n", prop.multiProcessorCount,
         (prop.memoryBusWidth / 8.0f * prop.memoryClockRate * 2.0f) / 1e6f);

  const int Q_SIZE = NUM_QUERIES * HEAD_DIM;
  const int K_SIZE = HEAD_DIM;
  const int S_SIZE = NUM_QUERIES;

  std::vector<float> h_Q(Q_SIZE), h_K(K_SIZE);
  std::vector<float> h_scores_cpu(S_SIZE), h_scores_gpu(S_SIZE);

  for (int i = 0; i < Q_SIZE; i++)
    h_Q[i] = (i % 17 - 8) * 0.01f;
  for (int i = 0; i < K_SIZE; i++)
    h_K[i] = (i % 13 - 6) * 0.01f;

  float scale = 1.0f / sqrtf((float)HEAD_DIM);
  cpu_attention_scores(h_Q, h_K, h_scores_cpu, scale);

  float *d_Q, *d_K, *d_scores;
  cudaMalloc(&d_Q, Q_SIZE * sizeof(float));
  cudaMalloc(&d_K, K_SIZE * sizeof(float));
  cudaMalloc(&d_scores, S_SIZE * sizeof(float));

  cudaMemcpy(d_Q, h_Q.data(), Q_SIZE * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_K, h_K.data(), K_SIZE * sizeof(float), cudaMemcpyHostToDevice);

  dim3 grid(NUM_QUERIES / WARPS_PER_BLOCK); // 128 blocks
  dim3 block(BLOCK_THREADS);                // 256 threads

  // Warmup
  attention_score_kernel<<<grid, block>>>(d_Q, d_K, d_scores, scale);
  cudaDeviceSynchronize();

  // Correctness
  cudaMemcpy(h_scores_gpu.data(), d_scores, S_SIZE * sizeof(float),
             cudaMemcpyDeviceToHost);

  int mismatches = 0;
  float max_err = 0.0f;
  for (int i = 0; i < S_SIZE; i++) {
    float err = fabsf(h_scores_gpu[i] - h_scores_cpu[i]);
    max_err = fmaxf(max_err, err);
    if (err > 1e-4f)
      mismatches++;
  }
  printf("Correctness: %s (max_err=%.6f, mismatches=%d/%d)\n\n",
         mismatches == 0 ? "PASS" : "FAIL", max_err, mismatches, S_SIZE);

  for (int i = 0; i < 8; i++)
    printf("  score[%d]: GPU=%.6f  CPU=%.6f\n", i, h_scores_gpu[i],
           h_scores_cpu[i]);

  // Timing
  const int ITERS = 1000;
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  for (int i = 0; i < 10; i++)
    attention_score_kernel<<<grid, block>>>(d_Q, d_K, d_scores, scale);
  cudaDeviceSynchronize();

  nvtxRangePush("attention_score_kernel");
  cudaEventRecord(start);
  for (int i = 0; i < ITERS; i++)
    attention_score_kernel<<<grid, block>>>(d_Q, d_K, d_scores, scale);
  cudaEventRecord(stop);
  nvtxRangePop();

  cudaEventSynchronize(stop);
  float elapsed_ms;
  cudaEventElapsedTime(&elapsed_ms, start, stop);
  float avg_us = (elapsed_ms / ITERS) * 1000.0f;

  float bytes = (Q_SIZE + K_SIZE + S_SIZE) * sizeof(float);
  float flops = NUM_QUERIES * HEAD_DIM * 2.0f;
  float peak_bw =
      (prop.memoryBusWidth / 8.0f * prop.memoryClockRate * 2.0f) / 1e6f;
  float achieved_bw = (bytes / 1e9f) / (avg_us / 1e6f);
  float bw_util = (achieved_bw / peak_bw) * 100.0f;

  printf("\n=== Performance ===\n");
  printf("  Avg time:        %.2f us\n", avg_us);
  printf("  Achieved BW:     %.2f GB/s\n", achieved_bw);
  printf("  Peak BW:         %.2f GB/s\n", peak_bw);
  printf("  BW utilization:  %.1f%%\n", bw_util);
  printf("  Arith intensity: %.3f FLOP/byte\n", flops / bytes);
  printf("  → Memory bandwidth bound (intensity < 1.0)\n\n");

  printf("=== Next Optimizations ===\n");
  printf("  1. Cache K in __shared__ (currently reloaded 8x per block)\n");
  printf("  2. float4 vectorized loads (4x bandwidth per transaction)\n");
  printf("  3. cp.async overlap → Module 3 TiledCopy\n");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_Q);
  cudaFree(d_K);
  cudaFree(d_scores);

  return 0;
}

/*
 * SELF-REVIEW QUESTIONS:
 *
 * Q1: Why __shfl_down_sync instead of __shared__ for warp reduction?
 * Q2: K is reloaded 8x per block — how would you fix this?
 * Q3: What is arithmetic intensity here and what does it mean?
 * Q4: If HEAD_DIM=64, what changes?
 * Q5: Why lane_id for local_partition instead of tid?
 */
