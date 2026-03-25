# Module 03: Load/Store - Memory Movement Patterns

## Learning Objectives

By the end of this module, you will:
- Use `ct.load()` and `ct.store()` correctly for 1D and 2D tiles
- Handle boundaries with padding for non-multiple shapes
- Implement a strided memory access pattern
- Understand gather/scatter concepts through index remapping

## Why Load/Store Matters

Most GPU kernels are limited by memory bandwidth, not arithmetic throughput.

A high-performance kernel must:
1. Read contiguous memory whenever possible
2. Reuse loaded data before loading more
3. Minimize uncoalesced or random access

`ct.load()` and `ct.store()` are your primary tools for this data movement.

## Core API

```python
# Load one tile from global memory
tile = ct.load(array, index=(block_id,), shape=(32,))

# Store one tile back to global memory
ct.store(output, index=(block_id,), tile=tile)
```

For boundary-safe kernels, use launch-time or kernel-level padding behavior so extra lanes do not read invalid memory.

## Exercises Overview

1. Basic copy kernel (sanity check)
2. Boundary-safe scaling kernel
3. 2D tile load/store kernel
4. Strided downsample kernel
5. Gather/scatter remap using an index map

## Run Tests

```bash
python test.py
```

## Next Module

Module 04 moves from memory movement to tile arithmetic and shape transforms.
