# Hints for ex04_move_vs_forward

## H1 — Concept Direction
`std::move` always produces an rvalue. `std::forward<T>` produces an lvalue if T is an lvalue reference, and an rvalue if T is not. In a template `T&&` parameter, T is deduced based on the argument's category.

## H2 — Names the Tool
Use `std::forward<T>(arg)` in `wrapper_forward` to preserve category. Use `std::move(arg)` in `wrapper_move` to always cast to rvalue. The difference: forward respects the original, move forces rvalue.

## H3 — Minimal Usage (Unrelated Context)
```cpp
template<typename T>
void forwarder(T&& x) {
    func(std::forward<T>(x));  // Preserves: lvalue→lvalue, rvalue→rvalue
}

template<typename T>
void mover(T&& x) {
    func(std::move(x));  // Always rvalue, even if x was lvalue
}
```
