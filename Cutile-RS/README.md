# CuTile-RS Tutorial

This tutorial is a fill-in-the-code path for learning CuTile Rust from memory fundamentals to Tensor Core-oriented kernels.

## Learning Flow

1. Read module `README.md`.
2. Implement `kernel.rs` by replacing `// FILL IN` sections.
3. Run tests from the module directory.
4. Compare against `solution.rs`.

Use reference solutions in tests:

```bash
TUTORIAL_USE_SOLUTION=1 rustc --test test.rs -o test_bin && ./test_bin
```

## Modules

- `00_setup`
- `01_data_model`
- `02_kernel_basics`
- `03_load_store`
- `04_tile_operations`
- `05_matrix_operations`
- `06_advanced_tiling`
- `07_reductions_atomics`
- `08_persistent_kernels`
- `09_capstone`
