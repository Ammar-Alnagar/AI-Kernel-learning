# H100 Fill-in GEMM Tutorial

Mirrors `kernels/gemm/educational_h100` with starter/solution split.

## Levels

- `01`: simple float GEMM baseline
- `02`: simple bf16 GEMM baseline
- `03`: shared memory tiling
- `04`: warp Tensor Core MMA (`warp::mma_AB`)
- `05`: warpgroup Tensor Core MMA (`warpgroup::mma_AB`)
- `06`: TMA + double buffering
- `07`: producer/consumer warp specialization
- `08`: multi-consumer warpgroup tiling

## Usage

```bash
make LEVEL=06 TRACK=starter clean
make LEVEL=06 TRACK=starter run
```

To inspect solution:

```bash
make LEVEL=06 TRACK=solution clean
make LEVEL=06 TRACK=solution run
```
