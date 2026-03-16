// CONCEPT: Variadic templates — tuple-like storage
// FORMAT: IMPLEMENT
// TIME_TARGET: 30 min
// WHY_THIS_MATTERS: Variadic templates enable flexible APIs. CUTLASS uses them for tile shapes.
// CUDA_CONNECTION: Kernel launch arguments as variadic pack — forward to device function.

#include <iostream>
#include <utility>
#include <tuple>

// TODO: Implement a simple Tuple class using variadic templates
// Requirements:
// 1. Template: template<typename... Ts> class Tuple
// 2. Inherit from TupleElement<size, T> for each type
// 3. Provide get<N>() method to access element by index
// 4. Provide sizeof...(Ts) for tuple size

// Helper: TupleElement stores a single value at a specific index
template<size_t I, typename T>
struct TupleElement {
    T value;
    TupleElement() = default;
    TupleElement(const T& v) : value(v) {}
    TupleElement(T&& v) : value(std::move(v)) {}
};

// TODO: Implement Tuple using variadic template parameter pack
// Hint: Use multiple inheritance from TupleElement<0, T0>, TupleElement<1, T1>, etc.
template<typename... Ts>
class Tuple;

// Base case: empty tuple
template<>
class Tuple<> {
public:
    static constexpr size_t size() { return 0; }
};

// Recursive case: Tuple<T, Rest...>
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
    
    // Get element by index
    template<size_t I>
    auto& get() {
        if constexpr (I == 0) {
            return TupleElement<0, T>::value;
        } else {
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

// Helper function to create tuple (like std::make_tuple)
template<typename... Ts>
Tuple<Ts...> make_tuple(Ts&&... args) {
    return Tuple<Ts...>(std::forward<Ts>(args)...);
}

// TODO: Implement a function that prints tuple contents
// Use fold expression: (..., print_element(x))
template<typename TupleType, size_t... I>
void print_tuple_impl(const TupleType& t, std::index_sequence<I...>) {
    std::cout << "(";
    // Fold expression: print each element with comma separator
    ((std::cout << t.template get<I>() << (I == sizeof...(I) - 1 ? "" : ", ")), ...);
    std::cout << ")\n";
}

template<typename... Ts>
void print_tuple(const Tuple<Ts...>& t) {
    print_tuple_impl(t, std::index_sequence_for<Ts...>{});
}

// TODO: Implement a function that applies a function to each tuple element
// Use fold expression with unary right fold
template<typename Func, typename... Ts>
void for_each_in_tuple(const Tuple<Ts...>& t, Func&& f) {
    [&]<size_t... I>(std::index_sequence<I...>) {
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
    
    std::cout << "\n=== CUTLASS Mapping ===\n";
    std::cout << "CUTLASS uses variadic templates for:\n";
    std::cout << "  - Tile shapes: cutlass::gemm::GemmShape<M, N, K>\n";
    std::cout << "  - Kernel arguments: launch<Ts...>(args...)\n";
    std::cout << "  - Policy packs: template<typename... Policies>\n";
    
    return 0;
}

// VERIFY: Expected output:
// === Creating Tuple ===
// Tuple size: 3
//
// === Accessing elements ===
// get<0>: 42 (int)
// get<1>: 3.14 (double)
// get<2>: hello (string)
//
// === Printing tuple ===
// (42, 3.14, hello)
//
// === For each element ===
//   Element: 42 (type size: 4)
//   Element: 3.14 (type size: 8)
//   Element: hello (type size: 32)
//
// === Empty tuple ===
// Empty tuple size: 0
//
// === CUTLASS Mapping ===
// CUTLASS uses variadic templates for...
