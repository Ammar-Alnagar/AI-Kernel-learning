# Hints for ex03_sfinae_vs_concepts

## H1 — Concept Direction
SFINAE uses `std::enable_if_t<condition, int> = 0` as a second template parameter. If the condition is false, substitution fails and that overload is removed. Concepts use `template<typename T> requires Concept<T>` — much cleaner.

## H2 — Names the Tool
SFINAE: `template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>`. Concepts: `template<typename T> concept FloatingPoint = std::is_floating_point_v<T>;` then `template<typename T> requires FloatingPoint<T>`.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// SFINAE:
template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void foo(T x);

// Concepts (C++20):
template<typename T> requires std::integral<T>
void foo(T x);

// Or abbreviated:
void foo(std::integral auto x);
```
