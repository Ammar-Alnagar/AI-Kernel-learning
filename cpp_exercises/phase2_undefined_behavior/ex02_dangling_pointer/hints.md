# Hints for ex02_dangling_pointer

## H1 — Concept Direction
The function `get_dangling_pointer()` returns a pointer to memory owned by a local `DataHolder`. When the function returns, the local is destroyed and frees the memory. The returned pointer now points to freed memory — it's dangling.

## H2 — Names the Tool
Fix by returning the `DataHolder` itself (the owner), not the raw pointer. Use move semantics to transfer ownership: `return holder;` invokes the move constructor, transferring the pointer safely.

## H3 — Minimal Usage (Unrelated Context)
```cpp
// Wrong: returns dangling pointer
int* getData() {
    std::vector<int> v = {1, 2, 3};
    return v.data();  // UB: v destroyed, pointer dangles
}

// Right: return the owner
std::vector<int> getData() {
    return {1, 2, 3};  // Move semantics, safe
}
```
