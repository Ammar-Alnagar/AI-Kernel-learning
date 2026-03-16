# ex05: Data Race on Non-Atomic Variable (UB)

Debug undefined behavior caused by unsynchronized concurrent access to a shared variable.

## What You Build

A fix for a multi-threaded counter that loses updates due to data races, using `std::atomic<int>`.

## What You Observe

The buggy version produces different results each run (lost updates, non-deterministic). ThreadSanitizer reports data race errors. The fixed version with `std::atomic` produces consistent, correct results (400000 = 4 threads × 100000 iterations).

## CUTLASS/CUDA Mapping

Device-side atomics (`atomicAdd`, `atomicCAS`) provide the same guarantees as `std::atomic`. CUTLASS uses atomics in epilogue fusion for reduction operations. Within a warp, `__syncwarp()` can replace atomics for shared memory (cheaper).

## Build Command

```bash
# Buggy version (ThreadSanitizer catches data race)
g++ -std=c++20 -O2 -fsanitize=thread -o ex05_buggy exercise.cpp -lpthread && ./ex05_buggy

# Fixed version
g++ -std=c++20 -O2 -o ex05_fixed solution.cpp -lpthread && ./ex05_fixed
```
