# Nsight Compute Workflow (Reference)

```bash
ncu --set full --section SpeedOfLight ./target/release/ex01_cuda_gemm_bench
ncu --metrics smsp__inst_executed_pipe_tensor.sum ./target/release/ex01_cuda_gemm_bench
ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed ./target/release/ex01_cuda_gemm_bench
```

Interpretation:
- High tensor-pipe activity + high SM utilization => compute-heavy path.
- Low tensor-pipe + high DRAM utilization => memory pressure dominates.
