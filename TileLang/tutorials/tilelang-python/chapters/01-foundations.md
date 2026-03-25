# Chapter 1: Foundations

## Concept Brief
You will learn TileLang kernel anatomy and the execution model:
- Program shape and launch configuration
- Basic tensor indexing
- Elementwise and reduction patterns

## Exercises
- `exercises/ch01_elementwise.py`
- `exercises/ch01_reduction.py`

## Fill-in Tasks
1. Define kernel signature and tensor shapes.
2. Map program/thread IDs to data indices.
3. Add boundary guards.
4. Implement writeback logic.

## Checkpoint
Run `checkpoints/test_ch01.py` and ensure:
- Numerical correctness against NumPy/PyTorch reference
- No out-of-bounds behavior

## Performance Challenge
Beat the provided Python baseline by a measurable margin.
Record speedup and effective bandwidth.

## Reflection Prompt
Which parts of your kernel map directly to hardware execution, and which are pure algorithmic logic?
