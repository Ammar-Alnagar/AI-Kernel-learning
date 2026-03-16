# Hints for ex05_specialization_dispatch

## H1 — Concept Direction
Full specialization provides a custom implementation for a specific type. Syntax: `template<> struct Name<SpecificType> { ... };`. The primary template is the fallback; specializations override for specific types.

## H2 — Names the Tool
Specialize `KernelConfig` for each CUDA type: `template<> struct KernelConfig<float>`, `template<> struct KernelConfig<__half>`, `template<> struct KernelConfig<__nv_bfloat16>`. Each returns different `threads_per_block()`.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Primary template:
template<typename T>
struct Config { static int value() { return 0; } };

// Specialization for int:
template<>
struct Config<int> { static int value() { return 42; } };

Config<float>::value();  // Returns 0 (primary)
Config<int>::value();    // Returns 42 (specialization)
```
