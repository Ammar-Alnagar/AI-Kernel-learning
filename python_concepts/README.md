# Python for ML Inference & Kernel Engineering

Speedrun Python for GPU kernel work — nothing more.

## Prerequisites

- Systems-level thinking (you already have this)
- Basic familiarity with CUDA concepts

## What's Covered

| Module | Topic | Kernel Engineering Relevance |
|--------|-------|------------------------------|
| 01 | Functions & Decorators | `@torch.compile`, `@cutlass.jit`, higher-order kernel launchers |
| 02 | Comprehensions | Building kernel config grids, tensor shape tuples |
| 03 | Dataclasses | Kernel launch configs, tensor metadata structs |
| 04 | Context Managers | CUDA streams, `torch.cuda.device`, resource cleanup |
| 05 | Type Hints | Tensor shape annotations, IDE support for kernel APIs |
| 06 | Tensor Indexing | Attention masks, KV cache slicing, broadcast dims |
| 07 | CLI & I/O | Benchmark scripts, path handling for model weights |
| 08 | Timing & Benchmarking | CUDA events, `ncu` profiling, benchmark harnesses |

## Capstone

A complete benchmark harness that:
- Parses CLI args (`--kernel`, `--dtype`, `--sizes`)
- Stores config in a `@dataclass`
- Times kernels with CUDA events
- Outputs formatted tables + CSV
- Invokes `ncu` for profiling

## Setup

```bash
python setup.py  # Validates env, prints Python version + torch info
```

## Workflow

1. Complete each `_FILL_IN.py` exercise (10–20 min each)
2. Predict behavior before running
3. Answer checkpoint questions after running
4. Check `solutions/` only if stuck

## Rules

- No OOP. No inheritance. No `__init__`. No class methods.
- Only `@dataclass` as plain structs.
- No asyncio/threading/multiprocessing internals.
- Every exercise ties to real kernel work.
