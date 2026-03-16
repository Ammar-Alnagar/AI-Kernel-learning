# ex03: SFINAE vs Concepts

Compare pre-C++20 SFINAE with C++20 concepts for template constraints.

## What You Build

Two sets of overloaded functions: one using SFINAE (`std::enable_if_t`) and one using C++20 concepts. Both constrain templates to floating-point or integral types.

## What You Observe

Both approaches achieve the same result — only matching overloads are considered. Concepts produce cleaner syntax and better error messages. CUTLASS 2.x uses SFINAE; CUTLASS 3.x uses concepts.

## CUTLASS/CUDA Mapping

CUTLASS 2.x: `template<typename T, typename = std::enable_if_t<is_numeric<T>>>`. CUTLASS 3.x: `template<typename T> requires cutlass::is_numeric<T>`. Reading CUTLASS requires fluency in both patterns.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex03 exercise.cpp && ./ex03
```
