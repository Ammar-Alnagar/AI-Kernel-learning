# ex01: Torn Reads/Writes — 64-bit Counter

Debug undefined behavior caused by non-atomic 64-bit operations.

## What You Build

A fix for a multi-threaded 64-bit counter that exhibits torn reads (values that were never written) due to non-atomic operations.

## What You Observe

The buggy version produces inconsistent results with ThreadSanitizer reporting data races. The fixed version with `std::atomic<uint64_t>` produces consistent, correct results.

## CUTLASS/CUDA Mapping

Device-side 64-bit counters need `atomicAdd((unsigned long long*)ptr, 1)`. Without atomics, multiple threads updating a global counter get torn results. CUTLASS uses atomics for reduction operations that accumulate 64-bit values.

## Build Command

```bash
# Buggy version (ThreadSanitizer catches data race)
g++ -std=c++20 -O2 -fsanitize=thread -o ex01_buggy exercise.cpp -lpthread && ./ex01_buggy

# Fixed version
g++ -std=c++20 -O2 -o ex01_fixed solution.cpp -lpthread && ./ex01_fixed
```
