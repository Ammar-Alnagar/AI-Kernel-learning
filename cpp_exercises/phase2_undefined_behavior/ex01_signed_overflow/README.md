# ex01: Signed Integer Overflow (UB)

Debug undefined behavior caused by signed integer overflow.

## What You Build

A fix for a summation function that overflows when accumulating near INT32_MAX.

## What You Observe

The buggy version triggers sanitizer errors and produces inconsistent results between builds. The fixed version uses int64_t for accumulation, avoiding overflow entirely.

## CUTLASS/CUDA Mapping

CUTLASS uses `size_t` and `uint32_t` for tile indices and loop counters. Signed overflow in kernel loops would cause infinite loops or skipped tiles. Unsigned wrap is defined and safe for index arithmetic.

## Build Command

```bash
# Buggy version (sanitizer catches UB)
g++ -std=c++20 -O2 -fsanitize=undefined -o ex01_buggy exercise.cpp && ./ex01_buggy

# Fixed version
g++ -std=c++20 -O2 -o ex01_fixed solution.cpp && ./ex01_fixed
```
