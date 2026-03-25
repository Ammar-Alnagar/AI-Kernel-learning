# Chapter 3: Performance Optimization

## Concept Brief
Focus on hardware-aware speedups:
- Coalesced memory access
- Latency hiding via pipelining
- Occupancy and register pressure tradeoffs

## Exercises
- `exercises/ch03_coalescing.py`
- `exercises/ch03_pipeline.py`

## Fill-in Tasks
1. Reorder accesses to improve coalescing.
2. Add software pipeline stages.
3. Tune unroll and tile sizes.
4. Compare throughput before/after each change.

## Checkpoint
Run `checkpoints/test_ch03.py` and ensure:
- Correct outputs
- Consistent timing methodology

## Performance Challenge
Achieve a target speedup vs Chapter 2 baseline.

## Reflection Prompt
Which optimization produced the biggest gain, and what profiler metric confirmed it?
