# ex04: std::move vs std::forward

Understand when to use each by observing their different behaviors.

## What You Build

Two template wrappers: one that forwards (preserves category) and one that always moves (forces rvalue).

## What You Observe

`wrapper_forward(named)` calls the lvalue overload and leaves `named` unchanged. `wrapper_move(named)` calls the rvalue overload and empties `named` (unintended move). This shows why forward is for templates, move is for explicit stealing.

## CUTLASS/CUDA Mapping

CUTLASS kernel launchers use perfect forwarding to accept both lvalue and rvalue arguments: `template<typename T> void launch(T&& args)`. Using `std::move` here would break lvalue arguments. Using `std::forward` preserves the caller's intent.

## Build Command

```bash
g++ -std=c++20 -O2 -Wall -Wextra -o ex04 exercise.cpp && ./ex04
```
