# Hints for ex03_uniqueptr_move_only

## H1 — Concept Direction
unique_ptr has a deleted copy constructor but a valid move constructor. The error "call to deleted constructor" means you're trying to COPY, not MOVE. Use `std::move()` to invoke the move constructor.

## H2 — Names the Tool
`std::move(ptr)` casts `ptr` to an rvalue, enabling the move constructor. After `ptr2 = std::move(ptr1)`, ptr1 is guaranteed to be `nullptr` by the C++ standard.

## H3 — Minimal Usage (Unrelated Context)
```cpp
std::unique_ptr<int> p1 = std::make_unique<int>(42);
std::unique_ptr<int> p2 = std::move(p1);  // OK: moves
// std::unique_ptr<int> p3 = p1;          // ERROR: copy deleted
if (p1) { /* Won't execute — p1 is nullptr */ }
if (p2) { /* Executes — p2 owns the int */ }
```
