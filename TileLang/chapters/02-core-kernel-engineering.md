# Chapter 2: Core Kernel Engineering

## Concept Brief
Move from simple kernels to structured tiling:
- Block/tile decomposition
- Shared-memory staging
- Synchronization for correctness

## Exercises
- `exercises/ch02_tiled_copy.py`
- `exercises/ch02_shared_memory.py`

## Fill-in Tasks
1. Choose tile sizes for target tensor shape.
2. Implement cooperative loads into shared memory.
3. Add synchronization barriers at correct points.
4. Validate correctness across non-divisible dimensions.

## Checkpoint
Run `checkpoints/test_ch02.py` and verify:
- Correct output for edge-case shapes
- Stable results with random seeds

## Performance Challenge
Compare global-memory-only vs shared-memory approach and report the gap.

## Reflection Prompt
What tile size tradeoffs did you observe between occupancy and memory reuse?
