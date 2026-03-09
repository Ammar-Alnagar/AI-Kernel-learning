#include <cuda_runtime.h>
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <iostream>
#include <nvtx3/nvToolsExt.h>

using namespace cute;

__global__ void vectorized_copy_kernel(float *gmem_data, float *gmem_out) {
  constexpr int M = 64;
  constexpr int N = 64;

  auto gmem_layout = make_layout(make_shape(Int<M>{}, Int<N>{}));
  auto gmem_tensor = make_tensor(make_gmem_ptr(gmem_data), gmem_layout);
  auto gmem_out_tensor = make_tensor(make_gmem_ptr(gmem_out), gmem_layout);

  __shared__ float smem_static[M * N];
  auto smem_tensor = make_tensor(make_smem_ptr(smem_static), gmem_layout);

  // TODO 1: Define Copy_Atom for 128-bit transfer over float elements
  using CopyAtom = /* YOUR CODE HERE */;

  // TODO 2: Build tiled_copy — atom + thread layout (256 threads) + value
  // layout
  auto tiled_copy = /* YOUR CODE HERE */;

  // TODO 3: Get this thread's slice
  auto thr_copy = /* YOUR CODE HERE */;

  // TODO 4: Partition and copy gmem -> smem
  /* YOUR CODE HERE */
  copy(tiled_copy, src, dst);

  __syncthreads();

  // TODO 5: Partition and copy smem -> gmem_out
  /* YOUR CODE HERE */
  copy(tiled_copy, src2, dst2);
}

int main() {
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("=== Vectorized 128-bit Copy ===\nGPU: %s\n", prop.name);

  constexpr int M = 64, N = 64;
  constexpr int SIZE = M * N;
  constexpr size_t BYTES = SIZE * sizeof(float);

  float *d_in, *d_out;
  cudaMalloc(&d_in, BYTES);
  cudaMalloc(&d_out, BYTES);

  std::vector<float> h_data(SIZE);
  for (int i = 0; i < SIZE; i++)
    h_data[i] = static_cast<float>(i);
  cudaMemcpy(d_in, h_data.data(), BYTES, cudaMemcpyHostToDevice);

  for (int i = 0; i < 10; i++)
    vectorized_copy_kernel<<<1, 256>>>(d_in, d_out);
  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  constexpr int NUM_ITER = 100;
  nvtxRangePush("vectorized_copy_kernel");
  cudaEventRecord(start);
  for (int i = 0; i < NUM_ITER; i++)
    vectorized_copy_kernel<<<1, 256>>>(d_in, d_out);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  nvtxRangePop();

  float elapsed_ms;
  // TODO 6: Fix elapsed time calculation (one call, two events)
  /* YOUR CODE HERE */
  elapsed_ms /= NUM_ITER;

  // TODO 7: Bandwidth formula — remember to account for both read AND write
  float bandwidth = /* YOUR CODE HERE */;

  printf("Average time: %.3f ms\nAchieved bandwidth: %.1f GB/s\n", elapsed_ms,
         bandwidth);

  std::vector<float> h_out(SIZE);
  cudaMemcpy(h_out.data(), d_out, BYTES, cudaMemcpyDeviceToHost);

  bool pass = true;
  for (int i = 0; i < SIZE; i++) {
    if (h_out[i] != static_cast<float>(i)) {
      pass = false;
      printf("Mismatch at %d: expected %f got %f\n", i, (float)i, h_out[i]);
      break;
    }
  }
  printf("\n[%s] Vectorized copy verified\n", pass ? "PASS" : "FAIL");

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaFree(d_in);
  cudaFree(d_out);
  return pass ? 0 : 1;
}
