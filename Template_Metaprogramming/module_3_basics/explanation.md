# Module 3: Template Metaprogramming Basics

## Overview
This module introduces the fundamentals of template metaprogramming, focusing on compile-time computation and type manipulation that form the basis for advanced libraries like CUTLASS.

## Learning Objectives
By the end of this module, students will be able to:
- Distinguish between compile-time and runtime computation
- Implement template recursion for compile-time calculations
- Use type traits and std::enable_if effectively
- Apply conditional compilation with templates
- Understand value-dependent and type-dependent expressions
- Work with template template parameters
- Implement basic expression SFINAE

## Topic 1: Compile-Time vs Runtime

Template metaprogramming enables computation at compile time, which can improve runtime performance.

### Compile-Time Calculations
```cpp
// Compile-time factorial calculation
template<int N>
struct Factorial {
    static constexpr int value = N * Factorial<N-1>::value;
};

// Base case specialization
template<>
struct Factorial<0> {
    static constexpr int value = 1;
};

// Usage
constexpr int fact5 = Factorial<5>::value;  // Computed at compile time
static_assert(fact5 == 120, "Factorial of 5 should be 120");
```

### Compile-Time vs Runtime Comparison
```cpp
#include <chrono>
#include <iostream>

// Runtime calculation
int runtime_factorial(int n) {
    if (n <= 1) return 1;
    return n * runtime_factorial(n - 1);
}

// Compile-time calculation
template<int N>
struct CompileTimeFactorial {
    static constexpr int value = N * CompileTimeFactorial<N-1>::value;
};

template<>
struct CompileTimeFactorial<0> {
    static constexpr int value = 1;
};

int main() {
    // Runtime calculation
    auto start = std::chrono::high_resolution_clock::now();
    int rt_result = runtime_factorial(10);
    auto end = std::chrono::high_resolution_clock::now();
    
    // Compile-time calculation (instantaneous at runtime)
    constexpr int ct_result = CompileTimeFactorial<10>::value;
    
    std::cout << "Runtime result: " << rt_result << std::endl;
    std::cout << "Compile-time result: " << ct_result << std::endl;
    
    return 0;
}
```

## Topic 2: Template Recursion

Template recursion is a fundamental technique in template metaprogramming.

### Recursive Template Structures
```cpp
// Compile-time Fibonacci sequence
template<int N>
struct Fibonacci {
    static constexpr int value = Fibonacci<N-1>::value + Fibonacci<N-2>::value;
};

// Base cases
template<>
struct Fibonacci<0> {
    static constexpr int value = 0;
};

template<>
struct Fibonacci<1> {
    static constexpr int value = 1;
};

// Usage
constexpr int fib10 = Fibonacci<10>::value;  // 55
```

### Recursive Template Functions
```cpp
// Sum of numbers from 1 to N at compile time
template<int N>
struct SumToN {
    static constexpr int value = N + SumToN<N-1>::value;
};

template<>
struct SumToN<1> {
    static constexpr int value = 1;
};

// Alternative with function templates
template<int N>
constexpr int sum_to_n() {
    if constexpr (N <= 1) {
        return N;
    } else {
        return N + sum_to_n<N-1>();
    }
}
```

## Topic 3: Type Traits and std::enable_if

Type traits enable conditional compilation based on type properties.

### Standard Library Type Traits
```cpp
#include <type_traits>

// Check if type is integral
template<typename T>
void process_integral(T value) {
    static_assert(std::is_integral_v<T>, "T must be an integral type");
    std::cout << "Processing integral: " << value << std::endl;
}

// Check if type is floating point
template<typename T>
void process_floating(T value) {
    static_assert(std::is_floating_point_v<T>, "T must be a floating point type");
    std::cout << "Processing float: " << value << std::endl;
}

// Check if types are the same
template<typename T, typename U>
void compare_types() {
    if constexpr (std::is_same_v<T, U>) {
        std::cout << "Types are the same" << std::endl;
    } else {
        std::cout << "Types are different" << std::endl;
    }
}
```

### Custom Type Traits
```cpp
// Check if type has begin() and end() methods
template<typename T, typename = void>
struct is_iterable : std::false_type {};

template<typename T>
struct is_iterable<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

// Check if type is callable
template<typename T, typename = void>
struct is_callable : std::false_type {};

template<typename T>
struct is_callable<T, std::void_t<
    decltype(std::declval<T>()(std::declval<int>()))
>> : std::true_type {};
```

### Using std::enable_if
```cpp
#include <type_traits>

// Method 1: Using std::enable_if in return type
template<typename T>
typename std::enable_if<std::is_arithmetic<T>::value, T>::type
abs_value(T val) {
    return val < 0 ? -val : val;
}

// Method 2: Using std::enable_if as a template parameter
template<typename T,
         typename = std::enable_if_t<std::is_arithmetic_v<T>>>
T abs_value_v2(T val) {
    return val < 0 ? -val : val;
}

// Method 3: Using SFINAE with function parameters
template<typename T>
void process_number(T val, std::enable_if_t<std::is_integral_v<T>, int> = 0) {
    std::cout << "Processing integer: " << val << std::endl;
}

template<typename T>
void process_number(T val, std::enable_if_t<std::is_floating_point_v<T>, int> = 0) {
    std::cout << "Processing float: " << val << std::endl;
}
```

## Topic 4: Conditional Compilation with Templates

Templates enable conditional compilation based on types or values known at compile time.

### if constexpr (C++17)
```cpp
#include <type_traits>

template<typename T>
void conditional_process(T value) {
    if constexpr (std::is_integral_v<T>) {
        std::cout << "Processing integer: " << value << std::endl;
    } else if constexpr (std::is_floating_point_v<T>) {
        std::cout << "Processing float: " << value << std::endl;
    } else {
        std::cout << "Processing other type" << std::endl;
    }
}

// Complex conditional processing
template<typename Container>
auto get_size(Container& c) {
    if constexpr (requires { c.size(); }) {
        return c.size();
    } else if constexpr (requires { c.length(); }) {
        return c.length();
    } else {
        static_assert(false, "Container must have size() or length() method");
    }
}
```

### Traditional SFINAE Approach
```cpp
// Using SFINAE for conditional compilation
template<typename T>
auto process_container(T& container) -> decltype(container.size(), void()) {
    std::cout << "Size-based processing: " << container.size() << std::endl;
}

template<typename T>
auto process_container(T& container) -> decltype(container.length(), void()) {
    std::cout << "Length-based processing: " << container.length() << std::endl;
}
```

## Topic 5: Value-Dependent and Type-Dependent Expressions

Understanding dependent names is crucial in template metaprogramming.

### Dependent Names
```cpp
template<typename T>
struct Base {
    static constexpr int value = 42;
    using type = double;
};

template<typename T>
struct Derived : Base<T> {
    // Without 'typename', the compiler doesn't know 'type' is a type
    typename Base<T>::type member_var;
    
    // Without 'template', the compiler doesn't know 'func' is a template
    template<typename U>
    void call_base_func(U u) {
        this->template func<U>(u);  // 'template' keyword needed
    }
    
    void access_base_value() {
        // 'this->' needed to access dependent base members
        int val = this->value;  // Same as Base<T>::value
    }
};
```

### Template Template Parameters
```cpp
// Template template parameter example
template<template<typename> typename Container, typename T>
class GenericWrapper {
    Container<T> storage;
public:
    void add(const T& item) {
        storage.push_back(item);
    }
    
    size_t size() const {
        return storage.size();
    }
};

// Usage
using IntVectorWrapper = GenericWrapper<std::vector, int>;
using IntDequeWrapper = GenericWrapper<std::deque, int>;
```

## Topic 6: Expression SFINAE

Expression SFINAE allows checking if expressions are valid at compile time.

### Basic Expression SFINAE
```cpp
#include <type_traits>

// Check if a type has a serialize method
template<typename T>
class HasSerialize {
    template<typename U>
    static auto test(int) -> decltype(std::declval<U>().serialize(), std::true_type{});
    
    template<typename>
    static std::false_type test(...);
    
public:
    static constexpr bool value = decltype(test<T>(0))::value;
};

// Modern C++17 approach using std::void_t
template<typename T, typename = void>
struct has_serialize_method : std::false_type {};

template<typename T>
struct has_serialize_method<T, std::void_t<
    decltype(std::declval<T>().serialize())
>> : std::true_type {};
```

### Advanced Expression SFINAE
```cpp
// Check if type supports arithmetic operations
template<typename T>
struct supports_arithmetic {
    template<typename U>
    static auto test_add(int) -> decltype(
        std::declval<U>() + std::declval<U>(), 
        std::true_type{}
    );
    
    template<typename>
    static std::false_type test_add(...);
    
    template<typename U>
    static auto test_subtract(int) -> decltype(
        std::declval<U>() - std::declval<U>(), 
        std::true_type{}
    );
    
    template<typename>
    static std::false_type test_subtract(...);
    
    static constexpr bool has_add = decltype(test_add<T>(0))::value;
    static constexpr bool has_subtract = decltype(test_subtract<T>(0))::value;
    static constexpr bool value = has_add && has_subtract;
};
```

## Hands-on Exercises

### Exercise 1: Compile-Time Prime Checker
Create a template struct that determines if a number is prime at compile time.

```cpp
// TODO: Implement a compile-time prime checker
// Requirements:
// 1. Should use template recursion
// 2. Should be computed at compile time
// 3. Should work for reasonable values (up to at least 100)
// 4. Use template specialization for base cases
```

### Exercise 2: Type Trait for Container Detection
Create type traits that detect if a type is a container-like type.

```cpp
// TODO: Create type traits to detect:
// 1. If a type has begin()/end() methods
// 2. If a type has size() method
// 3. If a type is a container (has both begin/end and size)
// Use expression SFINAE or std::void_t approach
```

### Exercise 3: Conditional Function Dispatcher
Create a function that dispatches to different implementations based on type properties.

```cpp
// TODO: Create a dispatcher function that:
// 1. Takes any type T
// 2. If T is arithmetic, calls arithmetic_handler()
// 3. If T is a pointer, calls pointer_handler()
// 4. If T is a class with serialize() method, calls serializable_handler()
// 5. Otherwise, static_assert with error message
```

## Solutions to Exercises

### Solution 1: Compile-Time Prime Checker
```cpp
template<int N, int Divisor = N-1>
struct IsPrime {
    static constexpr bool value = 
        (N % Divisor != 0) && IsPrime<N, Divisor-1>::value;
};

// Base cases
template<int N>
struct IsPrime<N, 1> {
    static constexpr bool value = true;
};

template<int N>
struct IsPrime<N, 0> {
    static constexpr bool value = false;
};

// Special cases
template<>
struct IsPrime<0> {
    static constexpr bool value = false;
};

template<>
struct IsPrime<1> {
    static constexpr bool value = false;
};

template<>
struct IsPrime<2> {
    static constexpr bool value = true;
};

// Helper variable template
template<int N>
inline constexpr bool is_prime_v = IsPrime<N>::value;

// Usage
static_assert(is_prime_v<2>, "2 should be prime");
static_assert(is_prime_v<17>, "17 should be prime");
static_assert(!is_prime_v<4>, "4 should not be prime");
```

### Solution 2: Type Trait for Container Detection
```cpp
#include <type_traits>

// Check if type has begin() and end() methods
template<typename T, typename = void>
struct has_begin_end : std::false_type {};

template<typename T>
struct has_begin_end<T, std::void_t<
    decltype(std::declval<T>().begin()),
    decltype(std::declval<T>().end())
>> : std::true_type {};

// Check if type has size() method
template<typename T, typename = void>
struct has_size : std::false_type {};

template<typename T>
struct has_size<T, std::void_t<
    decltype(std::declval<T>().size())
>> : std::true_type {};

// Check if type is container-like (has begin/end and size)
template<typename T>
struct is_container_like {
    static constexpr bool value = 
        has_begin_end<T>::value && has_size<T>::value;
};

// Convenience variable templates
template<typename T>
inline constexpr bool has_begin_end_v = has_begin_end<T>::value;

template<typename T>
inline constexpr bool has_size_v = has_size<T>::value;

template<typename T>
inline constexpr bool is_container_like_v = is_container_like<T>::value;
```

### Solution 3: Conditional Function Dispatcher
```cpp
#include <type_traits>

// Handler functions
template<typename T>
void arithmetic_handler(T value) {
    std::cout << "Handling arithmetic: " << value << std::endl;
}

template<typename T>
void pointer_handler(T ptr) {
    if (ptr) {
        std::cout << "Handling pointer: " << *ptr << std::endl;
    } else {
        std::cout << "Handling null pointer" << std::endl;
    }
}

template<typename T>
void serializable_handler(T& obj) {
    std::cout << "Handling serializable object" << std::endl;
    obj.serialize();  // This should compile if T has serialize method
}

// Main dispatcher
template<typename T>
void dispatch_handler(T value) {
    if constexpr (std::is_arithmetic_v<T>) {
        arithmetic_handler(value);
    } else if constexpr (std::is_pointer_v<T>) {
        pointer_handler(value);
    } else if constexpr (has_serialize_method<T>::value) {
        serializable_handler(value);
    } else {
        static_assert(std::is_void_v<T>, "Unsupported type for dispatch");
    }
}

// Alternative implementation using SFINAE
template<typename T>
auto dispatch_handler_sfinae(T value) -> std::enable_if_t<std::is_arithmetic_v<T>> {
    arithmetic_handler(value);
}

template<typename T>
auto dispatch_handler_sfinae(T value) -> std::enable_if_t<std::is_pointer_v<T>> {
    pointer_handler(value);
}

template<typename T>
auto dispatch_handler_sfinae(T& value) -> std::enable_if_t<has_serialize_method<T>::value> {
    serializable_handler(value);
}
```

## Advanced Topic: Template Metaprogramming Patterns

### The Enable-If Pattern
```cpp
// Common pattern for constraining templates
template<typename T>
std::enable_if_t<std::is_integral_v<T>, T>
safe_divide(T a, T b) {
    if (b == 0) {
        throw std::invalid_argument("Division by zero");
    }
    return a / b;
}
```

### The Tag Dispatching Pattern
```cpp
#include <iterator>

struct random_access_tag {};
struct bidirectional_tag {};

template<typename Iterator>
void advance_impl(Iterator& it, int n, random_access_tag) {
    it += n;  // Efficient random access
}

template<typename Iterator>
void advance_impl(Iterator& it, int n, bidirectional_tag) {
    if (n >= 0) {
        while (n--) ++it;
    } else {
        while (n++) --it;
    }
}

template<typename Iterator>
void advance(Iterator& it, int n) {
    using category = typename std::iterator_traits<Iterator>::iterator_category;
    advance_impl(it, n, category{});
}
```

## Quiz Questions

1. What is the difference between compile-time and runtime computation in the context of template metaprogramming?

2. Explain the purpose of template specialization in recursive template structures.

3. What is SFINAE and why is it important in template metaprogramming?

4. How does `std::enable_if` work and when should it be used?

5. What are dependent names and why do they require special handling in templates?

## Summary
Module 3 introduced the fundamentals of template metaprogramming, covering compile-time computation, template recursion, type traits, conditional compilation, and expression SFINAE. These concepts are essential for understanding advanced template metaprogramming techniques used in high-performance libraries like CUTLASS.