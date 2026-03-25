"""Chapter 5 Exercise: Benchmark and tuning harness.

Build a reproducible benchmark runner for TileLang kernels.
"""

from dataclasses import dataclass
from time import perf_counter


@dataclass
class BenchmarkResult:
    name: str
    mean_ms: float
    p50_ms: float
    p95_ms: float
    stdev_ms: float
    throughput: float


def time_callable(fn, warmup: int = 20, iters: int = 100):
    # TODO:
    # 1) Run warmup calls
    # 2) Record per-iteration latency (ms)
    # 3) Return full sample list
    raise NotImplementedError


def summarize(name: str, samples_ms, flops: float):
    # TODO: compute mean/p50/p95/std and throughput
    raise NotImplementedError


def autotune(configs, build_and_run):
    # TODO:
    # 1) Loop through config candidates
    # 2) Benchmark each config
    # 3) Return best config and complete results table
    raise NotImplementedError
