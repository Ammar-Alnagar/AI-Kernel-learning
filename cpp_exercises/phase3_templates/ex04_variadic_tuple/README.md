# ex04: Variadic Templates — Tuple Implementation

Implement a tuple-like class using variadic templates and fold expressions.

## What You Build

A `Tuple<Ts...>` class that stores any number of heterogeneous values, with `get<N>()` access, `size()`, and helper functions using fold expressions.

## What You Observe

The tuple recursively inherits from `TupleElement<I, T>` for each type. Fold expressions enable concise iteration: `(..., print(get<I>()))`. Empty tuples (`Tuple<>`) are the base case.

## CUTLASS/CUDA Mapping

CUTLASS uses variadic templates for tile shapes (`GemmShape<M, N, K>`), kernel argument packs, and policy composition. Understanding variadic expansion is essential for reading CUTLASS kernel launchers.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex04 exercise.cpp && ./ex04
```
