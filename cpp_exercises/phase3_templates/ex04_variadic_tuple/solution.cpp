// SOLUTION: ex04_variadic_tuple
// Complete implementation of variadic template tuple

#include <iostream>
#include <utility>
#include <tuple>
#include <string>

// Helper: TupleElement stores a single value at a specific index
template<size_t I, typename T>
struct TupleElement {
    T value;
    TupleElement() = default;
    TupleElement(const T& v) : value(v) {}
    TupleElement(T&& v) : value(std::move(v)) {}
};

// Base case: empty tuple
template<>
class Tuple<> {
public:
    static constexpr size_t size() { return 0; }
};

// Recursive case: Tuple<T, Rest...>
// Inherits from TupleElement<0, T> and Tuple<Rest...>
template<typename T, typename... Rest>
class Tuple<T, Rest...> : private TupleElement<0, T>, public Tuple<Rest...> {
public:
    // Constructor: forward arguments to base classes
    Tuple(const T& first, const Rest&... rest) 
        : TupleElement<0, T>(first), Tuple<Rest...>(rest...) {}
    
    Tuple(T&& first, Rest&&... rest) 
        : TupleElement<0, T>(std::move(first)), Tuple<Rest...>(std::forward<Rest>(rest)...) {}
    
    // Size: 1 + size of rest
    static constexpr size_t size() { return 1 + sizeof...(Rest); }
    
    // Get element by index using constexpr if
    template<size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return TupleElement<0, T>::value;
        } else {
            // Recursively get from base tuple, adjusting index
            return Tuple<Rest...>::template get<I - 1>();
        }
    }
    
    template<size_t I>
    const auto& get() const {
        if constexpr (I == 0) {
            return TupleElement<0, T>::value;
        } else {
            return Tuple<Rest...>::template get<I - 1>();
        }
    }
};

// Helper function to create tuple
template<typename... Ts>
Tuple<Ts...> make_tuple(Ts&&... args) {
    return Tuple<Ts...>(std::forward<Ts>(args)...);
}

// Print tuple using fold expression
template<typename TupleType, size_t... I>
void print_tuple_impl(const TupleType& t, std::index_sequence<I...>) {
    std::cout << "(";
    // Fold expression: ((cout << get<0>), (cout << get<1>), ...)
    ((std::cout << t.template get<I>() << (I == sizeof...(I) - 1 ? "" : ", ")), ...);
    std::cout << ")\n";
}

template<typename... Ts>
void print_tuple(const Tuple<Ts...>& t) {
    print_tuple_impl(t, std::index_sequence_for<Ts...>{});
}

// Apply function to each tuple element using fold expression
template<typename Func, typename... Ts>
void for_each_in_tuple(const Tuple<Ts...>& t, Func&& f) {
    // C++20 lambda with template parameters
    [&]<size_t... I>(std::index_sequence<I...>) {
        // Unary right fold: (f(get<0>()), f(get<1>()), ...)
        (f(t.template get<I>()), ...);
    }(std::index_sequence_for<Ts...>{});
}

int main() {
    std::cout << "=== Creating Tuple ===\n";
    auto t = make_tuple(42, 3.14, std::string("hello"));
    std::cout << "Tuple size: " << t.size() << "\n";
    
    std::cout << "\n=== Accessing elements ===\n";
    std::cout << "get<0>: " << t.get<0>() << " (int)\n";
    std::cout << "get<1>: " << t.get<1>() << " (double)\n";
    std::cout << "get<2>: " << t.get<2>() << " (string)\n";
    
    std::cout << "\n=== Printing tuple ===\n";
    print_tuple(t);
    
    std::cout << "\n=== For each element ===\n";
    for_each_in_tuple(t, [](const auto& x) {
        std::cout << "  Element: " << x << " (type size: " << sizeof(x) << ")\n";
    });
    
    std::cout << "\n=== Empty tuple ===\n";
    Tuple<> empty;
    std::cout << "Empty tuple size: " << empty.size() << "\n";
    
    std::cout << "\n=== Single element tuple ===\n";
    auto single = make_tuple(999);
    std::cout << "Single tuple size: " << single.size() << "\n";
    std::cout << "get<0>: " << single.get<0>() << "\n";
    
    std::cout << "\n=== CUTLASS Mapping ===\n";
    std::cout << "CUTLASS uses variadic templates for:\n";
    std::cout << "  - Tile shapes: cutlass::gemm::GemmShape<M, N, K>\n";
    std::cout << "  - Kernel arguments: launch<Ts...>(args...)\n";
    std::cout << "  - Policy packs: template<typename... Policies>\n";
    std::cout << "  - Parameter packs: cutlass::KernelArguments<Ts...>\n";
    
    return 0;
}

// KEY_INSIGHT:
// Variadic templates accept a parameter pack: template<typename... Ts>
// Ts... expands to zero or more types: <>, <int>, <int, float, string>, etc.
//
// Key operations:
// 1. sizeof...(Ts) — count types in pack
// 2. Ts... — expand pack (e.g., function arguments)
// 3. Fold expressions: (..., op(x)) — apply op to each element
//
// Recursive pattern:
// Tuple<T, Rest...> inherits from TupleElement<0, T> and Tuple<Rest...>
// Base case: Tuple<> (empty)
//
// CUTLASS mapping: Variadic templates appear in:
// - cutlass::gemm::GemmShape<M, N, K> — tile dimensions
// - Kernel launch argument packs
// - Policy composition (multiple policy classes combined)
