# Capstone Report Template

## Setup
- GPU:
- Driver/CUDA:
- TileLang version:
- Input shapes:

## Correctness
- Reference implementation:
- Error metric:
- Tolerance:

## Performance Table
| Step | Kernel | Latency (ms) | TFLOPS | Speedup vs Step1 |
|---|---|---:|---:|---:|
| 1 | Naive GEMM |  |  | 1.00x |
| 2 | Tiled GEMM |  |  |  |
| 3 | Shared-memory GEMM |  |  |  |
| 4 | Tensor Core GEMM |  |  |  |

## Profiler Evidence
- Key metrics used:
- Occupancy:
- Tensor Core utilization:
- Memory throughput:

## Optimization Notes
- Biggest bottleneck found:
- Most effective optimization:
- Tradeoffs accepted:

## Final Configuration
- Block sizes:
- Pipeline stages:
- Precision mode:
- Why this was chosen:
