# Chapter 4: Tensor Core Mastery

## Concept Brief
Use Tensor Cores effectively:
- WMMA/Tensor Core-friendly tile shapes
- Mixed precision strategy
- Accumulation accuracy considerations

## Exercises
- `exercises/ch04_tensorcore_gemm.py`
- `exercises/ch04_attention_block.py`

## Fill-in Tasks
1. Configure Tensor Core-compatible fragment/tile dimensions.
2. Implement load/compute/store pipeline.
3. Add mixed-precision input and higher-precision accumulate.
4. Validate numerical error bounds.

## Checkpoint
Run `checkpoints/test_ch04.py` and verify:
- Correctness tolerance for mixed precision
- Tensor Core path selected in profiler

## Performance Challenge
Reach a significant TFLOPS increase over non-Tensor-Core kernel.

## Reflection Prompt
How did data layout and precision choices impact both speed and numerical stability?
