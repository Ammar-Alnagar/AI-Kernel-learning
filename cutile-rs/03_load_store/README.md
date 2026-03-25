# Module 03: Load and Store

## Goal

Learn safe global-memory traffic patterns:

- masked `ct.load`
- masked `ct.store`
- out-of-bounds padding
- gather/scatter indexing

## Exercises

1. `masked_pad_load`: load with `other=0.0` when out of range.
2. `masked_store`: store only valid lanes.
3. `gather_1d`: indirect read via index buffer.
4. `scatter_add_1d`: indirect write with accumulation semantics.

## Run

```bash
rustc --test test.rs -o test_bin /&&/&& ./test_bin
TUTORIAL_USE_SOLUTION=1 rustc --test test.rs -o test_bin /&&/&& ./test_bin
```
