# Chapter 6: Capstone - GEMM Optimization Ladder

## Goal
Implement and optimize GEMM in five stages:
1. Naive GEMM
2. Tiled GEMM
3. Shared-memory optimized GEMM
4. Tensor Core GEMM
5. Benchmark and profiler report

## Files
- `exercises/capstone/step1_naive_gemm.py`
- `exercises/capstone/step2_tiled_gemm.py`
- `exercises/capstone/step3_shared_gemm.py`
- `exercises/capstone/step4_tensorcore_gemm.py`
- `exercises/capstone/step5_report_template.md`

## Deliverables
- Correctness checks against reference GEMM
- Performance table across all five stages
- Final write-up with tuning decisions and evidence

## Completion Criteria
- Numerical correctness within chosen tolerance
- Clear speedup trend across stages
- Profiler-backed explanation of final performance
