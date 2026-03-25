# Chapter 5: Production Workflow

## Concept Brief
Build a repeatable optimization process:
- Benchmark harness design
- Profiling and bottleneck localization
- Auto-tuning with reproducible configs

## Exercises
- `exercises/ch05_benchmark_harness.py`

## Fill-in Tasks
1. Add warmup and timed iterations.
2. Collect latency, throughput, and variance.
3. Sweep tile/config candidates.
4. Persist best config and metadata.

## Checkpoint
Run `checkpoints/test_ch05.py` and ensure:
- Benchmark script produces stable statistics
- Output table contains all required fields

## Performance Challenge
Find and justify a best configuration for one workload family.

## Reflection Prompt
What minimal set of metrics is enough to guide reliable kernel tuning decisions?
