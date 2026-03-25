# ThunderKittens Fill-in Tutorial Track

This directory mirrors the educational GEMM tutorial layout with explicit `starter` and `solution` code paths.

## Layout

- `gemm_h100/`: Hopper track (`level_01` to `level_08`)
- `gemm_b200/`: Blackwell track (`level_01` to `level_09`)

Each level has:

- `level_XX_starter.cu`: fill-in exercise file with `TODO(...)` markers
- `level_XX_solution.cu`: completed reference implementation

Each track also has:

- `Makefile`: same flow as existing educational kernels, plus `TRACK=starter|solution`
- `launch.cu`: benchmark/correctness harness

## Build and run

Example (H100 level 04 starter):

```bash
cd tutorials/fill_in/gemm_h100
make LEVEL=04 TRACK=starter clean
make LEVEL=04 TRACK=starter run
```

Compare with solution:

```bash
make LEVEL=04 TRACK=solution clean
make LEVEL=04 TRACK=solution run
```

## Recommended learning order

1. H100: `01 -> 08`
2. B200: `01 -> 09`
3. Refactor one solved level into `prototype::lcf`
