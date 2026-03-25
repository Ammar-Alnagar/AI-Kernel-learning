# Module 08: Persistent Kernels and Optimization Patterns

## Learning Objectives

By the end of this module, you will:
- Understand persistent kernel scheduling ideas
- Process multiple tiles per block in a loop
- Reduce kernel launch overhead for large workloads
- Compare throughput-oriented vs latency-oriented launch choices

## Concepts

A persistent kernel keeps each block alive longer and assigns multiple work items to it.

Pattern:
1. Start from block ID
2. Stride across global tile space by total number of blocks
3. Process multiple tiles per block

## Exercises

1. Persistent 1D scale kernel
2. Persistent affine kernel with tile loop
3. Compute tile work ranges on host
4. Compare with single-pass launch

## Run

```bash
python test.py
```
