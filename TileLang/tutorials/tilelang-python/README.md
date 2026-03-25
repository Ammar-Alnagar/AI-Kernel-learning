# TileLang Python Mastery Tutorial

## Goal
Create a complete, fill-in-the-code learning path for TileLang Python, from fundamentals to Tensor Core optimization.

## Audience
- Python developers with NumPy/PyTorch basics
- GPU beginners to intermediate practitioners
- Engineers focused on performance tuning

## Learning Outcomes
By the end of this tutorial, you should be able to:
- Write and debug TileLang kernels
- Use tiling and memory hierarchy effectively
- Optimize kernels with measurement-driven iteration
- Build Tensor Core kernels for GEMM-class workloads

## Structure
1. Foundations
2. Core Kernel Engineering
3. Performance Optimization
4. Tensor Core Mastery
5. Production Workflow
6. Capstone: GEMM Optimization Ladder

## How to Use This Tutorial
For each chapter:
1. Read the concept brief in `chapters/`
2. Complete TODO blocks in `exercises/`
3. Run the corresponding checkpoint in `checkpoints/`
4. Hit the chapter performance target
5. Record findings and tradeoffs

## File Map
- `chapters/`: concept and exercise instructions
- `exercises/`: starter code with TODO markers
- `checkpoints/`: correctness/performance checks
- `exercises/capstone/`: end-to-end GEMM progression

## Suggested Workflow
1. Finish correctness first.
2. Add one optimization at a time.
3. Benchmark after each change.
4. Use profiler evidence for next decisions.
5. Keep a short tuning log for reproducibility.
