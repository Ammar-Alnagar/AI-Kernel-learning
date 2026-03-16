# Hints for ex06_cutlass_annotate

## H1 — Concept Direction
This exercise tests your understanding of all template patterns from previous exercises. Each section corresponds to a pattern: type alias export (ex01), typename for dependent names (ex01), type traits (ex03), variadic templates (ex04).

## H2 — Names the Tool
`using Name = Name;` exports template parameters. `typename Policy::X` accesses dependent types. `sizeof...(Pack)` counts pack elements. `(f(args), ...)` is a fold expression that expands to `f(arg1), f(arg2), ...`.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Type alias export:
template<typename T>
struct Container { using value_type = T; };
using T = typename Container<float>::value_type;  // T is float

// Fold expression:
template<typename... Args>
void print_all(Args... args) {
    (std::cout << ... << args);  // Expands to: cout << arg1 << arg2 << ...
}
```
