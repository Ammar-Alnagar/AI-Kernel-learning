# Module 2: Template Fundamentals

## Overview
This module focuses on mastering basic template syntax and concepts that are essential for advanced template metaprogramming and CUTLASS usage.

## Learning Objectives
By the end of this module, students will be able to:
- Create and use class templates effectively
- Implement function templates with various parameter types
- Understand and apply template specialization
- Work with variadic templates
- Perform template argument deduction
- Apply perfect forwarding techniques

## Topic 1: Class Templates

Class templates allow creating generic classes that work with multiple types.

```cpp
#include <iostream>
#include <vector>

template<typename T>
class Stack {
private:
    std::vector<T> elements;

public:
    void push(const T& element) {
        elements.push_back(element);
    }

    void pop() {
        if (!elements.empty()) {
            elements.pop_back();
        }
    }

    const T& top() const {
        if (!elements.empty()) {
            return elements.back();
        }
        throw std::out_of_range("Stack is empty");
    }

    bool empty() const {
        return elements.empty();
    }

    size_t size() const {
        return elements.size();
    }
};

int main() {
    Stack<int> int_stack;
    Stack<std::string> string_stack;

    int_stack.push(42);
    string_stack.push("Hello");

    std::cout << int_stack.top() << std::endl;      // Outputs: 42
    std::cout << string_stack.top() << std::endl;   // Outputs: Hello

    return 0;
}
```

### Template Classes with Multiple Parameters

```cpp
template<typename Key, typename Value, typename Hash = std::hash<Key>>
class HashMap {
    // Implementation details...
};
```

### Template Classes with Non-Type Parameters

```cpp
template<typename T, size_t N>
class FixedArray {
private:
    T data[N];
    
public:
    T& operator[](size_t index) { return data[index]; }
    const T& operator[](size_t index) const { return data[index]; }
    size_t size() const { return N; }
};

int main() {
    FixedArray<int, 10> arr;  // Creates array of 10 integers
    arr[0] = 42;
    return 0;
}
```

## Topic 2: Function Templates

Function templates allow creating generic functions that work with multiple types.

```cpp
#include <type_traits>

// Basic function template
template<typename T>
T add(T a, T b) {
    return a + b;
}

// Function template with multiple type parameters
template<typename T, typename U>
auto multiply(T t, U u) -> decltype(t * u) {
    return t * u;
}

// C++14 and later: auto return type deduction
template<typename T, typename U>
auto multiply_v2(T t, U u) {
    return t * u;
}

// Function template with constraints (C++20 concepts)
#ifdef HAS_CONCEPTS
template<std::integral T>
T gcd(T a, T b) {
    while (b != 0) {
        T temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}
#endif
```

### Template Argument Deduction

The compiler can often deduce template arguments automatically:

```cpp
template<typename T>
void print_type_info(const T& value) {
    std::cout << "Value: " << value 
              << ", Type: " << typeid(T).name() << std::endl;
}

int main() {
    print_type_info(42);        // T is deduced as int
    print_type_info(3.14);      // T is deduced as double
    print_type_info("hello");   // T is deduced as const char (&)[6]
    
    // Explicit template argument specification
    print_type_info<int>(42.5); // T is explicitly int, value is converted
    
    return 0;
}
```

## Topic 3: Template Parameters

Templates can accept different kinds of parameters:

### Type Parameters
```cpp
template<typename T>  // or <class T>
class Container {
    T value;
};
```

### Non-Type Parameters
```cpp
template<int N>
class Buffer {
    char data[N];
};

template<typename T, int SIZE>
class StaticVector {
    T data[SIZE];
    int count = 0;
public:
    void push(const T& item) {
        if (count < SIZE) {
            data[count++] = item;
        }
    }
};
```

### Template Template Parameters
```cpp
template<template<typename> class Container, typename T>
class GenericProcessor {
    Container<T> container;
public:
    void add(const T& item) {
        container.push(item);
    }
};

// Usage
GenericProcessor<std::vector, int> processor;  // Uses std::vector<int>
```

## Topic 4: Template Specialization

Template specialization allows providing custom implementations for specific types.

### Full Specialization
```cpp
#include <cstring>

template<typename T>
class Serializer {
public:
    static void serialize(const T& obj, std::ostream& out) {
        // Generic serialization
        out << obj;
    }
};

// Full specialization for char*
template<>
class Serializer<char*> {
public:
    static void serialize(const char* str, std::ostream& out) {
        if (str) {
            out << "String: " << str;
        } else {
            out << "Null string";
        }
    }
};

// Full specialization for int
template<>
void serialize<int>(const int& value, std::ostream& out) {
    out << "Integer: " << value;
}
```

### Partial Specialization
Partial specialization is only available for class templates:

```cpp
#include <type_traits>

// Primary template
template<typename T, typename Enabled = void>
struct is_printable : std::false_type {};

// Partial specialization for types that support operator<<
template<typename T>
struct is_printable<T, std::void_t<decltype(std::declval<std::ostream&>() << std::declval<T>())>> 
    : std::true_type {};

// Another example: pointer types
template<typename T>
class Container {
    T value;
public:
    void print() { std::cout << "General: " << value << std::endl; }
};

template<typename T>
class Container<T*> {  // Partial specialization for pointer types
    T* ptr;
public:
    void print() { 
        if (ptr) {
            std::cout << "Pointer: " << *ptr << std::endl;
        } else {
            std::cout << "Null pointer" << std::endl;
        }
    }
};
```

## Topic 5: Variadic Templates

Variadic templates allow templates to accept a variable number of arguments.

### Basic Variadic Template
```cpp
#include <iostream>

// Base case (terminator)
template<typename T>
void print(T&& t) {
    std::cout << t << std::endl;
}

// Recursive case
template<typename T, typename... Args>
void print(T&& t, Args&&... args) {
    std::cout << t << " ";
    print(args...);  // Recursive call with remaining arguments
}

int main() {
    print(1, 2, 3, "hello", 4.5);
    // Output: 1 2 3 hello 4.5
    return 0;
}
```

### Variadic Class Template
```cpp
template<typename... Types>
class Tuple {};

// Specialization for single type
template<typename T>
class Tuple<T> {
    T value;
public:
    Tuple(const T& v) : value(v) {}
    T get() const { return value; }
};

// Recursive tuple
template<typename Head, typename... Tail>
class Tuple<Head, Tail...> {
    Head head;
    Tuple<Tail...> tail;
public:
    Tuple(Head h, Tail... t) : head(h), tail(t...) {}
    
    Head get_head() const { return head; }
    auto get_tail() const -> decltype(tail) { return tail; }
};
```

### Fold Expressions (C++17)
```cpp
template<typename... Args>
auto sum(Args... args) {
    return (args + ...);  // Unary right fold
}

template<typename... Args>
bool all_true(Args... args) {
    return (args && ...);  // Unary right fold for logical AND
}

template<typename... Args>
void print_fold(Args... args) {
    ((std::cout << args << " "), ...);  // Pack expansion with fold
    std::cout << std::endl;
}
```

## Topic 6: Perfect Forwarding

Perfect forwarding preserves the value category (lvalue/rvalue) of forwarded arguments.

```cpp
#include <utility>

template<typename T>
void wrapper(T&& arg) {
    // Forward arg preserving its value category
    some_function(std::forward<T>(arg));
}

// More complex example with multiple arguments
template<typename Func, typename... Args>
auto call_with_forwarding(Func&& func, Args&&... args) 
    -> decltype(func(std::forward<Args>(args)...)) {
    return func(std::forward<Args>(args)...);
}

// Example usage
class MyClass {
public:
    MyClass(int x, const std::string& s) { /* ... */ }
};

template<typename... Args>
std::unique_ptr<MyClass> make_my_class(Args&&... args) {
    return std::make_unique<MyClass>(std::forward<Args>(args)...);
}
```

## Hands-on Exercises

### Exercise 1: Generic Pair Class
Create a generic Pair class template that stores two values of potentially different types.

```cpp
// TODO: Implement a Pair class template
// Requirements:
// 1. Should store two values of potentially different types
// 2. Should provide accessors for both values
// 3. Should support copy and move operations
// 4. Should be usable with any types that support the required operations
```

### Exercise 2: Specialized Container
Create a generic container with specializations for specific types.

```cpp
template<typename T>
class OptimizedContainer {
    // TODO: Create a generic container
    // Then provide specializations for:
    // - bool: Use bit packing for space efficiency
    // - char*: Handle C-style strings specially
    // - std::string: Optimize for string operations
};
```

### Exercise 3: Variadic Function Calculator
Create a variadic template function that performs different operations based on the first argument.

```cpp
// TODO: Implement a calculator that takes an operation string
// followed by variable number of numeric arguments
// Example: calc("sum", 1, 2, 3, 4) -> 10
//          calc("product", 2, 3, 4) -> 24
//          calc("max", 5, 2, 8, 1) -> 8
```

## Solutions to Exercises

### Solution 1: Generic Pair Class
```cpp
template<typename T, typename U>
class Pair {
public:
    T first;
    U second;

    // Default constructor
    Pair() : first{}, second{} {}

    // Constructor with values
    Pair(const T& f, const U& s) : first(f), second(s) {}

    // Move constructor
    Pair(T&& f, U&& s) : first(std::forward<T>(f)), second(std::forward<U>(s)) {}

    // Copy constructor
    Pair(const Pair& other) : first(other.first), second(other.second) {}

    // Move constructor
    Pair(Pair&& other) noexcept : first(std::move(other.first)), second(std::move(other.second)) {}

    // Copy assignment
    Pair& operator=(const Pair& other) {
        if (this != &other) {
            first = other.first;
            second = other.second;
        }
        return *this;
    }

    // Move assignment
    Pair& operator=(Pair&& other) noexcept {
        if (this != &other) {
            first = std::move(other.first);
            second = std::move(other.second);
        }
        return *this;
    }
};
```

### Solution 2: Specialized Container
```cpp
#include <vector>
#include <string>
#include <cstring>

// Primary template
template<typename T>
class OptimizedContainer {
    std::vector<T> data;
public:
    void add(const T& item) { data.push_back(item); }
    size_t size() const { return data.size(); }
    const T& operator[](size_t idx) const { return data[idx]; }
};

// Specialization for bool (bit packing)
template<>
class OptimizedContainer<bool> {
    std::vector<uint8_t> data;
    size_t count;
    
public:
    OptimizedContainer() : count(0) {}
    
    void add(bool value) {
        size_t byte_idx = count / 8;
        size_t bit_idx = count % 8;
        
        if (byte_idx >= data.size()) {
            data.push_back(0);
        }
        
        if (value) {
            data[byte_idx] |= (1 << bit_idx);
        } else {
            data[byte_idx] &= ~(1 << bit_idx);
        }
        ++count;
    }
    
    bool operator[](size_t idx) const {
        size_t byte_idx = idx / 8;
        size_t bit_idx = idx % 8;
        return (data[byte_idx] >> bit_idx) & 1;
    }
    
    size_t size() const { return count; }
};

// Specialization for char*
template<>
class OptimizedContainer<char*> {
    std::vector<std::string> data;  // Store as strings internally
public:
    void add(char* str) {
        if (str) {
            data.emplace_back(str);
        } else {
            data.emplace_back("(null)");
        }
    }
    
    const char* operator[](size_t idx) const {
        return data[idx].c_str();
    }
    
    size_t size() const { return data.size(); }
};
```

### Solution 3: Variadic Function Calculator
```cpp
#include <algorithm>
#include <functional>

// Helper function for sum
template<typename... Args>
auto sum_calc(Args... args) {
    return (args + ...);
}

// Helper function for product
template<typename... Args>
auto product_calc(Args... args) {
    return (args * ...);
}

// Helper function for max
template<typename... Args>
auto max_calc(Args... args) {
    return std::max({args...});
}

// Main calculator function
template<typename... Args>
auto calc(const std::string& op, Args... args) {
    if (op == "sum") {
        return sum_calc(args...);
    } else if (op == "product") {
        return product_calc(args...);
    } else if (op == "max") {
        return max_calc(args...);
    } else if (op == "min") {
        return std::min({args...});
    } else {
        throw std::invalid_argument("Unknown operation: " + op);
    }
}
```

## Advanced Topic: Template Parameter Packs

Understanding how to work with parameter packs is crucial for advanced template metaprogramming:

```cpp
#include <tuple>

// Expanding parameter packs in different contexts
template<typename... Args>
void expand_examples(Args... args) {
    // Function call expansion
    some_function(args...);
    
    // Array initialization expansion
    int arr[] = {args...};
    
    // Base class list expansion
    // class Derived : public Base<Args>... {};
    
    // Member initializer list expansion
    // SomeClass obj{args...};
    
    // Template argument list expansion
    auto tup = std::make_tuple(args...);
}

// Counting parameter pack size
template<typename... Args>
constexpr size_t count_args(Args... args) {
    return sizeof...(Args);
}

// Using parameter pack with fold expressions (C++17)
template<typename... Args>
void print_all(Args... args) {
    (std::cout << ... << args) << std::endl;  // Left fold
}
```

## Quiz Questions

1. What is the difference between template type parameters and non-type parameters?

2. Explain the difference between full and partial template specialization.

3. What are parameter packs and how are they used in variadic templates?

4. Why is perfect forwarding important and how does `std::forward` work?

5. What is the difference between `sizeof...(Args)` and `sizeof...(args)` in a variadic template?

## Summary
Module 2 covered essential template fundamentals including class templates, function templates, template parameters, specialization, variadic templates, and perfect forwarding. These concepts form the core of advanced template metaprogramming and are essential for understanding CUTLASS's template-heavy architecture.