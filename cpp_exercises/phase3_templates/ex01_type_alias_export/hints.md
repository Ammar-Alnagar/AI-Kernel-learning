# Hints for ex01_type_alias_export

## H1 — Concept Direction
Type aliases inside class templates allow the outside world to query what type the template was instantiated with. CUTLASS uses this extensively — `TensorRef<T, Layout>` exports `value_type`, `index_type`, etc.

## H2 — Names the Tool
Inside the class: `using value_type = T;`. Outside, access with `typename`: `typename Tensor2D<float>::value_type`. The `typename` keyword is required for dependent names (names that depend on template parameters).

## H3 — Minimal Usage (Unrelated Context)
```cpp
template<typename T>
class Container {
public:
    using value_type = T;  // Export the type
};

// Query from outside:
using ElementType = typename Container<float>::value_type;
// ElementType is float
```
