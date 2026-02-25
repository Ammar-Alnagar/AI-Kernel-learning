# Project 08: Pipelined GEMM with Async Copy

## Objective

Implement GEMM using CUDA's async copy operations (cp.async) for overlapping memory transfers with computation. This project teaches:
- Async memory operations (Hopper/Ampere)
- Software pipelining
- Multi-stage pipelines
- Producer-consumer synchronization

## Theory

### Async Copy Operations

Modern GPUs support async memory operations that decouple issue from completion:

```cpp
// Issue async copy (returns immediately)
cp.async.cg.shared.global [dst], [src], size;

// Wait for async copies to complete
cp.async.wait_group 0;
```

### Software Pipelining

Overlap memory loads with computation using multiple pipeline stages:

```
Stage 0: Load A0, B0 → Compute C0 += A0×B0
Stage 1: Load A1, B1 → Compute C1 += A1×B1
Stage 2: Load A2, B2 → Compute C2 += A2×B2
         ... while Stage 0 computes
```

### Pipeline Stages

More stages = more parallelism but higher register pressure:
- 2-stage: Simple, low register usage
- 3-stage: Good balance
- 4+ stage: Maximum throughput, high register usage

## Your Task

### Step 1: Define Pipeline Structure

```cpp
constexpr int NumStages = 3;
__shared__ float As[NumStages][BM][BK];
__shared__ float Bs[NumStages][BK][BN];
```

### Step 2: Implement Pipeline Loop

```cpp
// Prologue: Load first stages
load_stage(stage[0]);
__syncthreads();

// Main loop
for (int k = 1; k < num_tiles; k++) {
    // Load next stage (async)
    cp_async_load(stage[k % NumStages]);
    
    // Compute previous stage
    __syncthreads();
    compute(stage[(k-1) % NumStages]);
    
    cp_async_commit();
}

// Epilogue: Final compute
__syncthreads();
compute(stage[(num_tiles-1) % NumStages]);
```

## Exercises

### Exercise 1: 2-Stage Pipeline (Required)
Implement basic double-buffering with async copy.

### Exercise 2: 3-Stage Pipeline (Challenge)
Add third stage for better latency hiding.

### Exercise 3: Async Wait Optimization (Advanced)
Use `cp.async.wait_group` instead of `__syncthreads()`.

---

**Ready to pipeline? Open `pipelined_gemm.cu`!**
