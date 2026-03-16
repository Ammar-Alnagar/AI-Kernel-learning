# Hints for ex03_use_after_move

## H1 — Concept Direction
After `std::move(buf1)`, buf1 is in a "valid but unspecified" state. In our implementation, `buf1.data` is set to `nullptr`. Accessing it (reading `data`) is undefined behavior — null pointer dereference.

## H2 — Names the Tool
Add an `isValid()` method that checks `data != nullptr`. Before accessing a potentially moved-from object, check `if (buf.isValid())`. Better: don't access moved-from objects at all.

## H3 — Minimal Usage (Unrelated Context)
```cpp
std::unique_ptr<int> p1 = std::make_unique<int>(42);
std::unique_ptr<int> p2 = std::move(p1);
// p1 is now nullptr — don't dereference!
if (p1) { /* Won't execute */ }
if (p2) { /* Executes — p2 owns the int */ }
```
