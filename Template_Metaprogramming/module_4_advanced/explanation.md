# Module 4: Advanced Template Metaprogramming Techniques

## Overview
This module delves into sophisticated template metaprogramming techniques used in high-performance libraries like CUTLASS, including design patterns, compile-time computations, and type manipulation.

## Learning Objectives
By the end of this module, students will be able to:
- Implement and recognize common TMP patterns (Enable-if, Tag dispatching, Policy-based design)
- Create expression templates for mathematical operations
- Perform complex compile-time computations
- Manipulate type lists and perform operations on them
- Implement higher-order template functions
- Use template aliasing effectively
- Understand and apply C++20 Concepts when available

## Topic 1: Template Metaprogramming Patterns

### Enable-If Pattern
The enable-if pattern is used to conditionally enable template functions based on type properties.

```cpp
#include <type_traits>

// Enable function only for integral types
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
multiply_safe(T a, T b) {
    // Safe multiplication for integers
    if (a != 0 && b > std::numeric_limits<T>::max() / a) {
        throw std::overflow_error("Multiplication overflow");
    }
    return a * b;
}

// Enable function only for floating-point types
template<typename T>
std::enable_if_t<std::is_floating_point_v<T>, T>
multiply_safe(T a, T b) {
    // Different logic for floating-point types
    return a * b;  // Floating-point handles overflow differently
}

// Multiple conditions
template<typename T>
std::enable_if_t<std::is_arithmetic_v<T> && sizeof(T) >= 4, T>
process_large_numbers(T value) {
    // Process only large arithmetic types
    return value * 2;
}
```

### Tag Dispatching Pattern
Tag dispatching selects different implementations based on type properties.

```cpp
#include <iterator>

// Tags for different iterator categories
struct random_access_tag {};
struct bidirectional_tag {};
struct forward_tag {};

// Implementation for random access iterators
template<typename Iterator>
void advance_impl(Iterator& it, int n, random_access_tag) {
    it += n;  // O(1) operation
}

// Implementation for bidirectional iterators
template<typename Iterator>
void advance_impl(Iterator& it, int n, bidirectional_tag) {
    if (n >= 0) {
        while (n--) ++it;
    } else {
        while (n++) --it;
    }
}

// Implementation for forward iterators
template<typename Iterator>
void advance_impl(Iterator& it, int n, forward_tag) {
    // Only forward movement allowed
    while (n-- > 0) ++it;
}

// Public interface that dispatches based on iterator category
template<typename Iterator>
void advance(Iterator& it, int n) {
    using category = typename std::iterator_traits<Iterator>::iterator_category;
    advance_impl(it, n, category{});
}
```

### Policy-Based Design Pattern
Policy-based design allows flexible composition of behaviors through template parameters.

```cpp
// Memory management policies
struct heap_policy {
    template<typename T>
    static T* allocate(size_t n) {
        return new T[n];
    }
    
    template<typename T>
    static void deallocate(T* ptr) {
        delete[] ptr;
    }
};

struct stack_policy {
    template<typename T>
    static T* allocate(size_t n) {
        static_assert(n <= 1024, "Stack allocation too large");
        thread_local static T buffer[1024];
        return buffer;
    }
    
    template<typename T>
    static void deallocate(T* ptr) {
        // Nothing to do for stack allocation
    }
};

// Threading policies
struct single_threaded_policy {
    template<typename Func>
    static void parallel_for(size_t start, size_t end, Func&& f) {
        for (size_t i = start; i < end; ++i) {
            f(i);
        }
    }
};

struct multi_threaded_policy {
    template<typename Func>
    static void parallel_for(size_t start, size_t end, Func&& f) {
        // Simplified parallel implementation
        #pragma omp parallel for
        for (size_t i = start; i < end; ++i) {
            f(i);
        }
    }
};

// Container that uses policies
template<typename T, 
         typename MemoryPolicy = heap_policy,
         typename ThreadingPolicy = single_threaded_policy>
class PolicyBasedContainer {
private:
    T* data;
    size_t capacity;
    size_t size_;

public:
    PolicyBasedContainer(size_t cap) : capacity(cap), size_(0) {
        data = MemoryPolicy::allocate<T>(capacity);
    }
    
    ~PolicyBasedContainer() {
        MemoryPolicy::deallocate(data);
    }
    
    void parallel_fill(const T& value) {
        ThreadingPolicy::parallel_for(0, size_, [this, &value](size_t i) {
            data[i] = value;
        });
    }
};
```

## Topic 2: Expression Templates

Expression templates enable lazy evaluation and eliminate temporary objects in mathematical expressions.

```cpp
#include <vector>
#include <iostream>

// Base expression template
template<typename Derived>
struct Expression {
    const Derived& impl() const { return static_cast<const Derived&>(*this); }
    
    template<typename T>
    auto operator+(const T& other) const {
        return AddExpr<Derived, T>(impl(), other);
    }
    
    template<typename T>
    auto operator*(const T& other) const {
        return MultExpr<Derived, T>(impl(), other);
    }
};

// Scalar value expression
template<typename T>
struct ScalarExpr : Expression<ScalarExpr<T>> {
    T value;
    
    ScalarExpr(T v) : value(v) {}
    
    auto operator[](size_t) const { return value; }
    size_t size() const { return 1; }
};

// Vector expression
template<typename T>
struct VectorExpr : Expression<VectorExpr<T>> {
    std::vector<T> data;
    
    VectorExpr(std::initializer_list<T> init) : data(init) {}
    VectorExpr(size_t size, T init_val = T{}) : data(size, init_val) {}
    
    auto operator[](size_t i) const { return data[i]; }
    auto operator[](size_t i) { return data[i]; }
    size_t size() const { return data.size(); }
};

// Addition expression
template<typename Left, typename Right>
struct AddExpr : Expression<AddExpr<Left, Right>> {
    const Left& left;
    const Right& right;
    
    AddExpr(const Left& l, const Right& r) : left(l), right(r) {}
    
    auto operator[](size_t i) const { 
        return left[i % left.size()] + right[i % right.size()]; 
    }
    
    size_t size() const { 
        return std::max(left.size(), right.size()); 
    }
};

// Multiplication expression
template<typename Left, typename Right>
struct MultExpr : Expression<MultExpr<Left, Right>> {
    const Left& left;
    const Right& right;
    
    MultExpr(const Left& l, const Right& r) : left(l), right(r) {}
    
    auto operator[](size_t i) const { 
        return left[i % left.size()] * right[i % right.size()]; 
    }
    
    size_t size() const { 
        return std::max(left.size(), right.size()); 
    }
};

// Assignment operator that evaluates expressions
template<typename T, typename Expr>
VectorExpr<T>& operator+=(VectorExpr<T>& vec, const Expression<Expr>& expr) {
    const auto& e = expr.impl();
    for (size_t i = 0; i < vec.size(); ++i) {
        vec[i] += e[i];
    }
    return vec;
}

// Evaluation function
template<typename Expr>
auto evaluate(const Expression<Expr>& expr) {
    const auto& e = expr.impl();
    VectorExpr<typename std::decay_t<decltype(e[0])>> result(e.size());
    for (size_t i = 0; i < e.size(); ++i) {
        result[i] = e[i];
    }
    return result;
}

// Usage example
void expression_template_example() {
    VectorExpr<double> v1{1.0, 2.0, 3.0};
    VectorExpr<double> v2{4.0, 5.0, 6.0};
    ScalarExpr<double> s(2.0);
    
    // This creates an expression tree, no computation yet
    auto expr = v1 + v2 * s;
    
    // Now the computation happens
    auto result = evaluate(expr);
    
    for (size_t i = 0; i < result.size(); ++i) {
        std::cout << result[i] << " ";  // Prints: 9 12 15
    }
    std::cout << std::endl;
}
```

## Topic 3: Compile-Time Computations

Advanced compile-time computations using templates.

### Compile-Time Algorithms
```cpp
// Compile-time greatest common divisor
template<int A, int B>
struct GCD {
    static constexpr int value = GCD<B, A % B>::value;
};

template<int A>
struct GCD<A, 0> {
    static constexpr int value = A;
};

// Compile-time least common multiple
template<int A, int B>
struct LCM {
    static constexpr int value = (A * B) / GCD<A, B>::value;
};

// Compile-time array reversal
template<typename T, size_t N, size_t... I>
constexpr std::array<T, N> reverse_array_impl(
    const std::array<T, N>& arr, 
    std::index_sequence<I...>) {
    return {{arr[N - 1 - I]...}};
}

template<typename T, size_t N>
constexpr auto reverse_array(const std::array<T, N>& arr) {
    return reverse_array_impl(arr, std::make_index_sequence<N>{});
}

// Compile-time string hashing
template<size_t N>
struct StrHash {
    static constexpr uint64_t hash(const char (&str)[N], uint64_t seed = 0xcbf29ce484222325ULL) {
        uint64_t hash = seed;
        for (size_t i = 0; i < N - 1; ++i) {
            hash ^= str[i];
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }
};
```

### Compile-Time Data Structures
```cpp
// Compile-time linked list
template<typename T>
struct Node {
    T value;
    template<typename... Ts>
    static constexpr auto create(Ts... vals) {
        return LinkedList<T, Ts...>{vals...};
    }
};

template<typename T, typename... Ts>
struct LinkedList {
    static constexpr std::array<T, sizeof...(Ts) + 1> values{T{}, Ts{}...};
    
    static constexpr size_t size() { return sizeof...(Ts) + 1; }
    
    template<size_t I>
    static constexpr T get() {
        static_assert(I < size(), "Index out of bounds");
        return values[I];
    }
};

// Compile-time type list
template<typename... Ts>
struct TypeList {
    static constexpr size_t size = sizeof...(Ts);
};

// Type list operations
template<typename List, typename T>
struct Append;

template<typename... Ts, typename T>
struct Append<TypeList<Ts...>, T> {
    using type = TypeList<Ts..., T>;
};

template<typename List, typename T>
using Append_t = typename Append<List, T>::type;

// Find type in type list
template<typename List, typename T>
struct Contains;

template<typename T>
struct Contains<TypeList<>, T> : std::false_type {};

template<typename U, typename... Ts, typename T>
struct Contains<TypeList<U, Ts...>, T> : Contains<TypeList<Ts...>, T> {};

template<typename T, typename... Ts>
struct Contains<TypeList<T, Ts...>, T> : std::true_type {};
```

## Topic 4: Type Lists and Operations

Type lists are fundamental in advanced template metaprogramming.

```cpp
// Basic type list
template<typename... Ts>
struct TypeList {};

// Length of type list
template<typename List>
struct Length;

template<typename... Ts>
struct Length<TypeList<Ts...>> {
    static constexpr size_t value = sizeof...(Ts);
};

// Front of type list
template<typename List>
struct Front;

template<typename T, typename... Ts>
struct Front<TypeList<T, Ts...>> {
    using type = T;
};

// Pop front from type list
template<typename List>
struct PopFront;

template<typename T, typename... Ts>
struct PopFront<TypeList<T, Ts...>> {
    using type = TypeList<Ts...>;
};

// Push back to type list
template<typename List, typename T>
struct PushBack;

template<typename... Ts, typename T>
struct PushBack<TypeList<Ts...>, T> {
    using type = TypeList<Ts..., T>;
};

// Transform operation on type list
template<typename List, template<typename> typename F>
struct Transform;

template<typename... Ts, template<typename> typename F>
struct Transform<TypeList<Ts...>, F> {
    using type = TypeList<F<Ts>...>;
};

// Filter operation on type list
template<typename List, template<typename> typename Predicate>
struct Filter;

template<template<typename> typename Predicate>
struct Filter<TypeList<>, Predicate> {
    using type = TypeList<>;
};

template<typename T, typename... Ts, template<typename> typename Predicate>
struct Filter<TypeList<T, Ts...>, Predicate> {
    using rest = typename Filter<TypeList<Ts...>, Predicate>::type;
    using type = std::conditional_t<
        Predicate<T>::value,
        typename PushBack<rest, T>::type,
        rest
    >;
};

// Example predicates
template<typename T>
struct IsIntegral {
    static constexpr bool value = std::is_integral_v<T>;
};

template<typename T>
struct IsFloatingPoint {
    static constexpr bool value = std::is_floating_point_v<T>;
};

// Usage examples
using MyTypes = TypeList<int, float, double, char, bool>;
using IntegralTypes = Filter<MyTypes, IsIntegral>::type;  // TypeList<int, char>
using TransformedTypes = Transform<IntegralTypes, std::add_const>::type;  // TypeList<const int, const char>
```

## Topic 5: Higher-Order Template Functions

Templates that operate on other templates.

```cpp
// Higher-order function: applies a template to each type in a type list
template<template<typename> typename F, typename... Ts>
constexpr auto apply_to_types(TypeList<Ts...>) {
    return TypeList<F<Ts>...>{};
}

// Higher-order function: combines two types
template<template<typename, typename> typename F, typename... Ts, typename... Us>
constexpr auto zip_with(TypeList<Ts...>, TypeList<Us...>) {
    static_assert(sizeof...(Ts) == sizeof...(Us), "Type lists must have same size");
    return TypeList<F<Ts, Us>...>{};
}

// Example: create pointer types for each type in list
template<typename T>
struct MakePointer {
    using type = T*;
};

// Example: pair two types together
template<typename T, typename U>
struct TypePair {
    using first = T;
    using second = U;
};

// Usage
using OriginalTypes = TypeList<int, float, double>;
using PointerTypes = decltype(apply_to_types<MakePointer>(OriginalTypes{}));
// Results in TypeList<int*, float*, double*>

using SecondTypes = TypeList<char, bool, long>;
using PairedTypes = decltype(zip_with<TypePair>(OriginalTypes{}, SecondTypes{}));
// Results in TypeList<TypePair<int, char>, TypePair<float, bool>, TypePair<double, long>>
```

## Topic 6: Template Aliasing and Type Manipulation

Using template aliases to simplify complex type manipulations.

```cpp
// Template aliases for complex types
template<typename T>
using Vec = std::vector<T>;

template<typename T>
using VecVec = std::vector<std::vector<T>>;

template<typename T>
using Identity = T;

// Conditional type aliasing
template<bool Condition, typename TrueType, typename FalseType>
using conditional_t = std::conditional_t<Condition, TrueType, FalseType>;

// Remove reference and const
template<typename T>
using remove_cref = std::remove_const_t<std::remove_reference_t<T>>;

// Type trait aliases
template<typename T>
using is_arithmetic = std::is_arithmetic<T>;

template<typename T>
inline constexpr bool is_arithmetic_v = std::is_arithmetic_v<T>;

// Complex type manipulation
template<typename T>
struct SmartTypeSelector {
    using type = conditional_t<
        std::is_integral_v<T>,
        int,
        conditional_t<
            std::is_floating_point_v<T>,
            double,
            std::string
        >
    >;
};

template<typename T>
using SmartType = typename SmartTypeSelector<T>::type;

// Alias templates for template template parameters
template<template<typename> typename Transform>
struct TypeTransformer {
    template<typename T>
    using apply = typename Transform<T>::type;
};
```

## Topic 7: C++20 Concepts (When Available)

Concepts provide a more readable way to constrain templates.

```cpp
#ifdef __cpp_concepts
#include <concepts>

// Define a concept for arithmetic types
template<typename T>
concept Arithmetic = std::is_arithmetic_v<T>;

// Define a concept for containers
template<typename T>
concept Container = requires(T t) {
    typename T::value_type;
    t.begin();
    t.end();
    t.size();
};

// Use concepts to constrain function templates
template<Arithmetic T>
T multiply_safe(T a, T b) {
    if (a != 0 && b > std::numeric_limits<T>::max() / a) {
        throw std::overflow_error("Multiplication overflow");
    }
    return a * b;
}

// More complex concept
template<typename T>
concept Iterable = requires(T t) {
    t.begin();
    t.end();
    requires std::input_iterator<decltype(t.begin())>;
};

// Function constrained by multiple concepts
template<Container C, Arithmetic T>
void fill_container(C& container, T value) {
    for (auto& elem : container) {
        elem = value;
    }
}

// Concept for callable objects
template<typename F, typename... Args>
concept Callable = requires(F&& f, Args&&... args) {
    std::invoke(std::forward<F>(f), std::forward<Args>(args)...);
};

// Use concept to create more expressive code
template<Callable<int, int> BinaryOp>
auto apply_op(BinaryOp&& op, int a, int b) {
    return op(a, b);
}
#endif
```

## Hands-on Exercises

### Exercise 1: Matrix Expression Templates
Extend the expression template example to support matrix operations.

```cpp
// TODO: Create expression templates for matrix operations
// Requirements:
// 1. Support matrix addition, subtraction, and scalar multiplication
// 2. Implement lazy evaluation to avoid temporary matrices
// 3. Support mixed operations (matrix + scalar, matrix * scalar)
// 4. Include proper size checking at compile or runtime
```

### Exercise 2: Type List Algorithms
Implement additional algorithms for type lists.

```cpp
// TODO: Implement the following type list operations:
// 1. Concatenate two type lists
// 2. Find the index of a type in a type list
// 3. Remove a type from a type list
// 4. Flatten nested type lists
// 5. Take first N types from a type list
```

### Exercise 3: Policy-Based Matrix Class
Create a matrix class that uses policy-based design for different storage and computation strategies.

```cpp
// TODO: Create a matrix class with policies for:
// 1. Storage policy (heap, stack, memory-mapped)
// 2. Computation policy (sequential, parallel, GPU)
// 3. Layout policy (row-major, column-major, blocked)
// 4. Include proper type constraints and error handling
```

## Solutions to Exercises

### Solution 1: Matrix Expression Templates
```cpp
#include <vector>

// Base matrix expression
template<typename Derived>
struct MatrixExpression {
    const Derived& impl() const { return static_cast<const Derived&>(*this); }
    
    size_t rows() const { return impl().rows(); }
    size_t cols() const { return impl().cols(); }
    
    template<typename Other>
    auto operator+(const MatrixExpression<Other>& other) const {
        return MatrixAddExpr<Derived, Other>(impl(), other.impl());
    }
    
    template<typename Scalar>
    auto operator*(const Scalar& scalar) const {
        return MatrixScalarMultExpr<Derived, Scalar>(impl(), scalar);
    }
};

// Dense matrix implementation
template<typename T>
struct DenseMatrix : MatrixExpression<DenseMatrix<T>> {
    std::vector<std::vector<T>> data;
    size_t rows_, cols_;
    
    DenseMatrix(size_t r, size_t c, T init_val = T{}) 
        : rows_(r), cols_(c), data(r, std::vector<T>(c, init_val)) {}
    
    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }
    
    T& operator()(size_t i, size_t j) { return data[i][j]; }
    const T& operator()(size_t i, size_t j) const { return data[i][j]; }
};

// Matrix addition expression
template<typename Left, typename Right>
struct MatrixAddExpr : MatrixExpression<MatrixAddExpr<Left, Right>> {
    const Left& left;
    const Right& right;
    
    MatrixAddExpr(const Left& l, const Right& r) : left(l), right(r) {
        if (l.rows() != r.rows() || l.cols() != r.cols()) {
            throw std::invalid_argument("Matrix dimensions don't match for addition");
        }
    }
    
    size_t rows() const { return left.rows(); }
    size_t cols() const { return left.cols(); }
    
    auto operator()(size_t i, size_t j) const { 
        return left(i, j) + right(i, j); 
    }
};

// Matrix-scalar multiplication expression
template<typename Matrix, typename Scalar>
struct MatrixScalarMultExpr : MatrixExpression<MatrixScalarMultExpr<Matrix, Scalar>> {
    const Matrix& matrix;
    const Scalar& scalar;
    
    MatrixScalarMultExpr(const Matrix& m, const Scalar& s) : matrix(m), scalar(s) {}
    
    size_t rows() const { return matrix.rows(); }
    size_t cols() const { return matrix.cols(); }
    
    auto operator()(size_t i, size_t j) const { 
        return matrix(i, j) * scalar; 
    }
};

// Evaluation function
template<typename T, typename Expr>
void evaluate_to(DenseMatrix<T>& result, const MatrixExpression<Expr>& expr) {
    const auto& e = expr.impl();
    if (result.rows() != e.rows() || result.cols() != e.cols()) {
        result = DenseMatrix<T>(e.rows(), e.cols());
    }
    
    for (size_t i = 0; i < e.rows(); ++i) {
        for (size_t j = 0; j < e.cols(); ++j) {
            result(i, j) = e(i, j);
        }
    }
}
```

### Solution 2: Type List Algorithms
```cpp
// Concatenate two type lists
template<typename List1, typename List2>
struct Concat;

template<typename... Ts, typename... Us>
struct Concat<TypeList<Ts...>, TypeList<Us...>> {
    using type = TypeList<Ts..., Us...>;
};

template<typename List1, typename List2>
using Concat_t = typename Concat<List1, List2>::type;

// Find index of type in type list
template<typename List, typename T>
struct IndexOf;

template<typename T, typename... Ts>
struct IndexOf<TypeList<T, Ts...>, T> {
    static constexpr size_t value = 0;
};

template<typename U, typename T, typename... Ts>
struct IndexOf<TypeList<U, Ts...>, T> {
    static constexpr size_t value = 1 + IndexOf<TypeList<Ts...>, T>::value;
};

template<typename T>
struct IndexOf<TypeList<>, T> {
    static constexpr size_t value = static_cast<size_t>(-1);  // Not found
};

// Remove first occurrence of type from type list
template<typename List, typename T>
struct Remove;

template<typename T, typename... Ts>
struct Remove<TypeList<T, Ts...>, T> {
    using type = TypeList<Ts...>;
};

template<typename U, typename T, typename... Ts>
struct Remove<TypeList<U, Ts...>, T> {
    using rest = typename Remove<TypeList<Ts...>, T>::type;
    using type = Append_t<TypeList<U>, typename rest::template Append<rest>>;
};

// Actually implement Remove properly:
template<typename T, typename... Ts>
struct RemoveImpl {
    using type = TypeList<Ts...>;
};

template<typename T, typename... Ts>
struct RemoveImpl<T, T, Ts...> {
    using type = TypeList<Ts...>;
};

template<typename T, typename U, typename... Ts>
struct RemoveImpl<T, U, Ts...> {
    using temp = typename RemoveImpl<T, Ts...>::type;
    using type = typename PushBack<temp, U>::type;
};

template<typename List, typename T>
struct Remove;

template<typename... Ts, typename T>
struct Remove<TypeList<Ts...>, T> {
    using type = typename RemoveImpl<T, Ts...>::type;
};

// Take first N types from type list
template<typename List, size_t N>
struct Take;

template<typename... Ts, size_t N>
struct Take<TypeList<Ts...>, N> {
    template<size_t I, typename... Taken, typename... Remaining>
    struct TakeHelper;
    
    template<size_t I, typename... Taken>
    struct TakeHelper<I, TypeList<Taken...>> {
        using type = TypeList<Taken...>;
    };
    
    template<size_t I, typename Head, typename... Tail, typename... Taken>
    struct TakeHelper<I, TypeList<Taken...>, Head, Tail...> {
        using type = typename std::conditional_t<
            I == 0,
            TypeList<Taken...>,
            typename TakeHelper<I-1, TypeList<Taken..., Head>, Tail...>::type
        >::type;
    };
    
    using type = typename TakeHelper<N, TypeList<>, Ts...>::type;
};

template<typename... Ts>
struct Take<TypeList<Ts...>, 0> {
    using type = TypeList<>;
};
```

### Solution 3: Policy-Based Matrix Class
```cpp
// Storage policies
struct HeapStoragePolicy {
    template<typename T>
    static std::vector<T> create_storage(size_t size) {
        return std::vector<T>(size);
    }
};

struct StackStoragePolicy {
    template<typename T>
    static auto create_storage(size_t size) {
        static_assert(size <= 1024, "Stack storage too large");
        thread_local static T storage[1024];
        return std::span<T>(storage, size);
    }
};

// Computation policies
struct SequentialComputationPolicy {
    template<typename Iterator, typename Func>
    static void for_each(Iterator begin, Iterator end, Func&& f) {
        for (auto it = begin; it != end; ++it) {
            f(*it);
        }
    }
};

struct ParallelComputationPolicy {
    template<typename Iterator, typename Func>
    static void for_each(Iterator begin, Iterator end, Func&& f) {
        #pragma omp parallel for
        for (auto it = begin; it != end; ++it) {
            f(*it);
        }
    }
};

// Layout policies
struct RowMajorLayout {
    size_t operator()(size_t rows, size_t cols, size_t i, size_t j) const {
        return i * cols + j;
    }
};

struct ColumnMajorLayout {
    size_t operator()(size_t rows, size_t cols, size_t i, size_t j) const {
        return j * rows + i;
    }
};

// Policy-based matrix
template<
    typename T,
    typename StoragePolicy = HeapStoragePolicy,
    typename ComputationPolicy = SequentialComputationPolicy,
    typename LayoutPolicy = RowMajorLayout
>
class PolicyMatrix {
private:
    std::vector<T> data_;
    size_t rows_, cols_;
    LayoutPolicy layout_;

public:
    PolicyMatrix(size_t rows, size_t cols) 
        : data_(StoragePolicy::create_storage<T>(rows * cols)),
          rows_(rows), cols_(cols), layout_() {}

    T& operator()(size_t i, size_t j) {
        return data_[layout_(rows_, cols_, i, j)];
    }

    const T& operator()(size_t i, size_t j) const {
        return data_[layout_(rows_, cols_, i, j)];
    }

    size_t rows() const { return rows_; }
    size_t cols() const { return cols_; }

    void fill(const T& value) {
        ComputationPolicy::for_each(data_.begin(), data_.end(),
            [&value](T& elem) { elem = value; });
    }
};
```

## Advanced Topic: Template Metaprogramming in CUTLASS Context

Understanding how these advanced techniques apply to CUTLASS:

```cpp
// Simplified example of CUTLASS-style template patterns
namespace cutlass_example {

// Configuration parameters as template parameters
template<
    typename Element,           // Data type (float, half, int8, etc.)
    int Alignment,             // Memory alignment
    typename Layout,           // Memory layout (row-major, column-major)
    int ElementsPerAccess      // Number of elements per memory access
>
struct AccessConfig {
    using element_type = Element;
    static constexpr int alignment = Alignment;
    using layout_type = Layout;
    static constexpr int elements_per_access = ElementsPerAccess;
};

// Operation traits
template<typename Element>
struct MultiplyAdd {
    CUTLASS_HOST_DEVICE
    Element operator()(const Element& a, const Element& b, const Element& c) const {
        return a * b + c;
    }
};

// Concept of an accumulator
template<typename T>
concept Accumulator = requires(T t, typename T::element_type a, typename T::element_type b) {
    typename T::element_type;
    t.accumulate(a, b);
    t.get();
};

// Example accumulator implementation
template<typename Element>
struct GemmAccumulator {
    using element_type = Element;
    Element value = Element{0};
    
    CUTLASS_HOST_DEVICE
    void accumulate(const Element& a, const Element& b) {
        value += a * b;
    }
    
    CUTLASS_HOST_DEVICE
    Element get() const { return value; }
};

} // namespace cutlass_example
```

## Quiz Questions

1. What is the main advantage of expression templates in numerical computing?

2. Explain the difference between tag dispatching and policy-based design.

3. How do type lists enable compile-time metaprogramming?

4. What is the purpose of higher-order template functions?

5. How do C++20 Concepts improve template code compared to SFINAE?

## Summary
Module 4 covered advanced template metaprogramming techniques including common patterns, expression templates, compile-time computations, type list operations, higher-order functions, and template aliasing. These techniques are fundamental to understanding and working with high-performance libraries like CUTLASS.