/**
 * @file 02_software_pipeline.cu
 * @brief Software Pipelining for GEMM
 * 
 * This tutorial demonstrates software pipelining:
 * - Overlapping memory and compute
 * - Multi-stage pipelines
 * - Producer-consumer pattern
 */

#include <cute/layout.hpp>
#include <cute/tensor.hpp>
#include <cute/int_tuple.hpp>
#include <cute/util/print.hpp>
#include <iostream>
#include <vector>
#include <cuda_runtime.h>

using namespace cute;

#define CUDA_CHECK(call) \
    do { cudaError_t err = call; if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl; exit(1); } \
    } while(0)

int main() {
    std::cout << "=== CuTe Tutorial: Software Pipelining ===" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Concept 1: What is Software Pipelining?
    // ================================================================
    std::cout << "--- Concept 1: Software Pipelining ---" << std::endl;
    std::cout << "Software pipelining overlaps memory operations with compute." << std::endl;
    std::cout << std::endl;
    std::cout << "Basic idea:" << std::endl;
    std::cout << "  - While computing tile K, load tile K+1" << std::endl;
    std::cout << "  - Hides memory latency behind computation" << std::endl;
    std::cout << "  - Increases instruction-level parallelism" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Without pipelining:" << std::endl;
    std::cout << "  [Load K=0] -> [Compute K=0] -> [Load K=1] -> [Compute K=1] -> ..." << std::endl;
    std::cout << std::endl;
    
    std::cout << "With pipelining:" << std::endl;
    std::cout << "  [Load K=0]" << std::endl;
    std::cout << "               [Compute K=0] -> [Load K=1]" << std::endl;
    std::cout << "                                 [Compute K=1] -> [Load K=2]" << std::endl;
    std::cout << "                                                   [Compute K=2] -> ..." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 1: Pipeline Stages
    // ================================================================
    std::cout << "--- Example 1: Pipeline Stages ---" << std::endl;
    
    const int NUM_STAGES = 3;
    
    std::cout << "Multi-stage pipeline with " << NUM_STAGES << " stages:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Stage 0: Load tile K (global -> shared)" << std::endl;
    std::cout << "Stage 1: MMA compute (shared -> registers)" << std::endl;
    std::cout << "Stage 2: Accumulate and store" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Each stage operates on different K-tiles simultaneously!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 2: Double Buffering
    // ================================================================
    std::cout << "--- Example 2: Double Buffering ---" << std::endl;
    
    const int PIPE_STAGES = 2;
    const int TILE_SIZE = 256;  // elements per tile
    
    std::cout << "Double buffering uses " << PIPE_STAGES << " buffers in shared memory." << std::endl;
    std::cout << std::endl;
    
    std::cout << "Buffer layout:" << std::endl;
    std::cout << "  Buffer 0: elements 0-" << (TILE_SIZE - 1) << std::endl;
    std::cout << "  Buffer 1: elements " << TILE_SIZE << "-" << (2 * TILE_SIZE - 1) << std::endl;
    std::cout << "  Total shared memory: " << (PIPE_STAGES * TILE_SIZE * 4) << " bytes" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Ping-pong pattern:" << std::endl;
    std::cout << "  Iteration 0: Load to Buffer 0, Compute from (previous)" << std::endl;
    std::cout << "  Iteration 1: Load to Buffer 1, Compute from Buffer 0" << std::endl;
    std::cout << "  Iteration 2: Load to Buffer 0, Compute from Buffer 1" << std::endl;
    std::cout << "  ..." << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 3: Pipeline Schedule
    // ================================================================
    std::cout << "--- Example 3: Pipeline Schedule ---" << std::endl;
    
    const int K_TILES = 8;
    
    std::cout << "Pipeline schedule for " << K_TILES << " K-tiles:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Time  Load    Compute" << std::endl;
    std::cout << "      Stage   Stage" << std::endl;
    std::cout << "----------------------" << std::endl;
    
    for (int t = 0; t < K_TILES + 1; ++t) {
        printf("T%-4d  ", t);
        if (t < K_TILES) {
            printf("Load K=%-2d", t);
        } else {
            printf("       ");
        }
        printf("  ");
        if (t > 0) {
            printf("Compute K=%d", t - 1);
        }
        printf("\n");
    }
    std::cout << std::endl;
    
    std::cout << "Notice: Load and Compute happen in parallel after pipeline fill!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 4: Producer-Consumer Pattern
    // ================================================================
    std::cout << "--- Example 4: Producer-Consumer Pattern ---" << std::endl;
    
    std::cout << "In GEMM, we have:" << std::endl;
    std::cout << "  Producer: Loads data from global to shared memory" << std::endl;
    std::cout << "  Consumer: Computes MMA using shared memory data" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Synchronization points:" << std::endl;
    std::cout << "  - After producer loads tile (before consumer reads)" << std::endl;
    std::cout << "  - After consumer finishes (before producer overwrites)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "With double buffering:" << std::endl;
    std::cout << "  - Producer and consumer work on different buffers" << std::endl;
    std::cout << "  - Reduces synchronization overhead" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 5: Register Pressure Considerations
    // ================================================================
    std::cout << "--- Example 5: Register Pressure ---" << std::endl;
    
    const int MMA_M = 16;
    const int MMA_N = 16;
    const int ACC_ELEMENTS = MMA_M * MMA_N;
    
    std::cout << "For a " << MMA_M << "x" << MMA_N << " MMA output:" << std::endl;
    std::cout << "  Accumulator registers: " << ACC_ELEMENTS << " (FP32)" << std::endl;
    std::cout << std::endl;
    
    std::cout << "With pipelining:" << std::endl;
    std::cout << "  - Need registers for current accumulator" << std::endl;
    std::cout << "  - Need registers for load addresses" << std::endl;
    std::cout << "  - May need registers for next tile setup" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Trade-off:" << std::endl;
    std::cout << "  - More stages = better latency hiding" << std::endl;
    std::cout << "  - More stages = higher register pressure" << std::endl;
    std::cout << "  - Higher register pressure = lower occupancy" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Sweet spot: Typically 2-4 stages for GEMM" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 6: Async Copy (Hopper/Ampere+)
    // ================================================================
    std::cout << "--- Example 6: Async Copy (Modern GPUs) ---" << std::endl;
    std::cout << "Newer GPUs support async copy operations:" << std::endl;
    std::cout << std::endl;
    
    std::cout << "CUDA Async Copy (Hopper):" << std::endl;
    std::cout << "  - cp.async.bulk instruction" << std::endl;
    std::cout << "  - Hardware-managed async copies" << std::endl;
    std::cout << "  - Reduces software overhead" << std::endl;
    std::cout << std::endl;
    
    std::cout << "Tensor Memory Accelerator (TMA):" << std::endl;
    std::cout << "  - Hardware tensor load/store" << std::endl;
    std::cout << "  - Automatic address calculation" << std::endl;
    std::cout << "  - Further reduces software overhead" << std::endl;
    std::cout << std::endl;
    
    std::cout << "CuTe provides abstractions for these features!" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Example 7: Pipeline Implementation Pattern
    // ================================================================
    std::cout << "--- Example 7: Pipeline Code Pattern ---" << std::endl;
    
    std::cout << "Pseudocode for pipelined GEMM:" << std::endl;
    std::cout << std::endl;
    std::cout << "  // Prologue: Load first tile" << std::endl;
    std::cout << "  load_tile(A, As[0], k = 0);" << std::endl;
    std::cout << "  load_tile(B, Bs[0], k = 0);" << std::endl;
    std::cout << "  __syncthreads();" << std::endl;
    std::cout << std::endl;
    std::cout << "  // Main loop" << std::endl;
    std::cout << "  for (int k = 0; k < K_tiles; ++k) {" << std::endl;
    std::cout << "    // Start loading next tile" << std::endl;
    std::cout << "    if (k + 1 < K_tiles) {" << std::endl;
    std::cout << "      load_tile(A, As[next], k + 1);" << std::endl;
    std::cout << "      load_tile(B, Bs[next], k + 1);" << std::endl;
    std::cout << "    }" << std::endl;
    std::cout << std::endl;
    std::cout << "    // Compute current tile" << std::endl;
    std::cout << "    __syncthreads();" << std::endl;
    std::cout << "    mma(As[curr], Bs[curr], D);" << std::endl;
    std::cout << std::endl;
    std::cout << "    // Swap buffers" << std::endl;
    std::cout << "    curr = next;" << std::endl;
    std::cout << "    next = 1 - next;" << std::endl;
    std::cout << "  }" << std::endl;
    std::cout << std::endl;

    // ================================================================
    // Summary
    // ================================================================
    std::cout << "=== Summary ===" << std::endl;
    std::cout << "1. Software pipelining overlaps memory and compute" << std::endl;
    std::cout << "2. Double buffering enables concurrent load/compute" << std::endl;
    std::cout << "3. Multi-stage pipelines hide latency better" << std::endl;
    std::cout << "4. Producer-consumer pattern with synchronization" << std::endl;
    std::cout << "5. Balance register pressure and latency hiding" << std::endl;
    std::cout << "6. Modern GPUs have hardware async copy support" << std::endl;
    std::cout << std::endl;

    return 0;
}
