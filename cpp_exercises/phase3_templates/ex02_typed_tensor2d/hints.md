# Hints for ex02_typed_tensor2d

## H1 — Concept Direction
The Tensor2D class needs two template parameters: the element type `T` and an index type (default `int32_t`). Export both as type aliases so generic code can query them.

## H2 — Names the Tool
Type aliases: `using value_type = T;` and `using index_type = IndexType;`. Access from outside: `typename Tensor2D<float>::value_type`. The `typename` keyword is required for dependent names.

## H3 — Minimal Usage (Unrelated Context)
```cpp
template<typename T, typename Index = int32_t>
class Array {
public:
    using value_type = T;
    using index_type = Index;
};

// Generic function:
template<typename ArrayType>
void process(ArrayType& a) {
    typename ArrayType::value_type v = 0;  // Works for any ArrayType
}
```
