# Hints for ex04_variadic_tuple

## H1 — Concept Direction
Variadic templates use `typename... Ts` to accept zero or more types. The pack `Ts...` can be expanded in function arguments, base classes, or fold expressions. Use recursion: `Tuple<T, Rest...>` inherits from `TupleElement<0, T>` and `Tuple<Rest...>`.

## H2 — Names the Tool
`sizeof...(Ts)` counts types in the pack. Fold expression: `(..., op(x))` applies `op` to each element. `std::index_sequence<I...>` generates compile-time indices for tuple access.

## H3 — Minimal Usage (Unrelated Context)
```cpp
template<typename... Ts>
void print_all(Ts... args) {
    ((std::cout << args << " "), ...);  // Fold expression
}
print_all(1, 2.0, "three");  // Prints: 1 2 three 

template<typename... Ts>
class Container : public Base<Ts>... {  // Pack expansion in base classes
};
```
