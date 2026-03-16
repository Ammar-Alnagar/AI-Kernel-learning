# ex06: CUTLASS Source Annotation

Read and annotate simplified CUTLASS 3.x source code patterns.

## What You Build

Answers to 9 questions about CUTLASS source patterns: type alias exports, dependent names, type traits, and fold expressions.

## What You Observe

CUTLASS source becomes readable once you recognize the patterns. Each "opaque" section uses concepts from previous exercises: type aliases, typename, specialization, variadic templates.

## CUTLASS/CUDA Mapping

This is real CUTLASS 3.x structure. `cutlass::gemm::collective::CollectiveGemm` uses exactly these patterns. Understanding them enables you to read and modify CUTLASS for custom kernels.

## Build Command

```bash
g++ -std=c++20 -O2 -o ex06 exercise.cpp && ./ex06
```
