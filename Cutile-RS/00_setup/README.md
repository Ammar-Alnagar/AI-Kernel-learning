# Module 00: Setup and First Kernel

## Goal

Start with the core execution model:

- one program instance handles one block of elements
- global index = `block_id * block_size + lane`
- launch grid = `ceil_div(n, block_size)`

## Exercises

1. `ceil_div`: compute blocks needed for `n` elements.
2. `vector_add_block`: implement one block body.
3. `vector_add_launch`: run all blocks.

## Run

```bash
rustc --test test.rs -o test_bin /&&/&& ./test_bin
TUTORIAL_USE_SOLUTION=1 rustc --test test.rs -o test_bin /&&/&& ./test_bin
```
