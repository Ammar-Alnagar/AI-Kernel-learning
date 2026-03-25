# Module 01: Data Model

## Goal

Understand how layout and types affect kernel behavior.

- flat indexing with explicit strides
- dtype conversion
- power-of-two constraints for tile sizes

## Exercises

1. `offset_2d`: compute linear offset from `(row, col)` and strides.
2. `gather_2d_strided`: read values using explicit stride math.
3. `cast_array`: convert dtype.
4. `is_power_of_two`: validate tile dimensions.
