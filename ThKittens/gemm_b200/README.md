# B200 Fill-in GEMM Tutorial

Mirrors `kernels/gemm/educational_b200` with starter/solution split.

## Levels

- `01`: simple float GEMM baseline
- `02`: simple bf16 GEMM baseline
- `03`: shared memory tiling
- `04`: warp Tensor Core MMA
- `05`: TMA global<->shared + warp MMA
- `06`: tcgen05 Tensor Core + TMA
- `07`: pipelined warp specialization
- `08`: epilogue pipelining
- `09`: cluster + warpgroup parallelism

## Usage

```bash
make LEVEL=08 TRACK=starter clean
make LEVEL=08 TRACK=starter run
```

Compare with solution:

```bash
make LEVEL=08 TRACK=solution clean
make LEVEL=08 TRACK=solution run
```
