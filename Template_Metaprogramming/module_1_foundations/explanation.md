# Module 1: Foundations of Modern C++

## Overview
This module establishes the solid foundation in advanced C++ concepts needed for template metaprogramming. We'll cover modern C++ features that form the building blocks for sophisticated template systems.

## Learning Objectives
By the end of this module, students will be able to:
- Use modern C++ features like auto, constexpr, and lambdas effectively
- Understand RAII and smart pointer patterns
- Apply move semantics appropriately
- Create basic function templates
- Perform type deduction with auto and decltype
- Understand SFINAE basics

## Topic 1: Modern C++ Features

### auto Keyword
The `auto` keyword allows the compiler to deduce variable types automatically.

```cpp
// Basic usage
auto x = 42;           // x is int
auto y = 42.5;         // y is double
auto z = "hello";      // z is const char*

// With complex types
std::vector<int> vec = {1, 2, 3, 4, 5};
auto it = vec.begin(); // it is std::vector<int>::iterator
```

### constexpr Functions
`constexpr` functions are evaluated at compile time when possible.

```cpp
constexpr int square(int x) {
    return x * x;
}

int main() {
    constexpr int result = square(5);  // Computed at compile time
    int arr[result];                   // Valid array size
    return 0;
}
```

### Lambda Functions
Lambda expressions allow creating anonymous functions inline.

```cpp
#include <algorithm>
#include <vector>

int main() {
    std::vector<int> vec = {5, 2, 8, 1, 9};

    // Sort in descending order
    std::sort(vec.begin(), vec.end(), [](int a, int b) {
        return a > b;
    });

    // Capture variables
    int multiplier = 10;
    auto multiply_by_ten = [multiplier](int x) {
        return x * multiplier;
    };

    return 0;
}
```

## Topic 2: RAII and Smart Pointers

### RAII (Resource Acquisition Is Initialization)
RAII is a C++ programming technique where resource management is tied to object lifetime.

```cpp
#include <memory>
#include <fstream>

class FileManager {
private:
    std::unique_ptr<std::ifstream> file;

public:
    FileManager(const std::string& filename) {
        file = std::make_unique<std::ifstream>(filename);
    }

    ~FileManager() {
        // File automatically closed when object goes out of scope
    }

    bool is_open() const {
        return file && file->is_open();
    }
};
```

### Smart Pointers
Modern C++ provides automatic memory management through smart pointers.

```cpp
#include <memory>

void smart_pointer_examples() {
    // unique_ptr - exclusive ownership
    auto ptr1 = std::make_unique<int>(42);
    
    // shared_ptr - shared ownership
    auto ptr2 = std::make_shared<int>(42);
    auto ptr3 = ptr2;  // Both ptr2 and ptr3 share ownership
    
    // weak_ptr - non-owning reference to shared_ptr
    std::weak_ptr<int> weak_ref = ptr2;
}
```

## Topic 3: Move Semantics and Rvalue References

### Lvalues and Rvalues
Understanding value categories is crucial for move semantics.

```cpp
int x = 42;        // x is an lvalue
int& lr = x;       // lr is an lvalue reference
int&& rr = 42;     // rr is an rvalue reference
int&& rr2 = x + 1; // x + 1 is an rvalue
```

### Move Constructor and Assignment
Move semantics allow efficient transfer of resources.

```cpp
#include <utility>
#include <vector>

class MyVector {
private:
    std::vector<int> data;

public:
    // Move constructor
    MyVector(MyVector&& other) noexcept : data(std::move(other.data)) {
        // other.data is now empty
    }

    // Move assignment operator
    MyVector& operator=(MyVector&& other) noexcept {
        if (this != &other) {
            data = std::move(other.data);
        }
        return *this;
    }
};
```

## Topic 4: Function Templates Basics

### Basic Function Template
Function templates allow writing generic functions that work with multiple types.

```cpp
template<typename T>
T max(T a, T b) {
    return (a > b) ? a : b;
}

int main() {
    int i = max(5, 10);        // Instantiated as max<int>
    double d = max(3.14, 2.71); // Instantiated as max<double>
    return 0;
}
```

### Multiple Template Parameters
Templates can have multiple type parameters.

```cpp
template<typename T, typename U>
auto multiply(T t, U u) -> decltype(t * u) {
    return t * u;
}

// C++14 and later can use auto return type deduction
template<typename T, typename U>
auto multiply_v2(T t, U u) {
    return t * u;
}
```

## Topic 5: Type Deduction

### auto and decltype
Understanding how the compiler deduces types is essential.

```cpp
#include <type_traits>

int main() {
    int x = 42;
    const int cx = x;
    const int& crx = x;

    auto a1 = x;        // a1 is int
    auto a2 = cx;       // a2 is int
    auto a3 = crx;      // a3 is int
    auto& a4 = x;       // a4 is int&
    auto& a5 = cx;      // a5 is const int&
    auto& a6 = crx;     // a6 is const int&

    decltype(x) b1 = 42;     // b1 is int
    decltype((x)) b2 = x;    // b2 is int& (parentheses matter!)
    
    return 0;
}
```

## Topic 6: SFINAE Basics

### Substitution Failure Is Not An Error
SFINAE is a fundamental concept in template metaprogramming.

```cpp
#include <type_traits>

// Enable function only for integral types
template<typename T>
typename std::enable_if<std::is_integral<T>::value, void>::type
process(T value) {
    // Process integral types
    printf("Processing integral: %d\n", static_cast<int>(value));
}

// Enable function only for floating-point types
template<typename T>
typename std::enable_if<std::is_floating_point<T>::value, void>::type
process(T value) {
    // Process floating-point types
    printf("Processing float: %f\n", static_cast<double>(value));
}
```

## Hands-on Exercises

### Exercise 1: Generic Min/Max Functions
Create generic min and max functions that work with any comparable type.

```cpp
// TODO: Implement generic min/max functions
// Requirements:
// 1. Should work with any type that supports comparison operators
// 2. Should return by value for primitive types
// 3. Should return by const reference for complex types
// 4. Include proper const-correctness
```

### Exercise 2: Smart Pointer Wrapper
Create a simple smart pointer class that implements RAII principles.

```cpp
template<typename T>
class SimplePtr {
    // TODO: Implement a basic smart pointer
    // Requirements:
    // 1. Store a raw pointer
    // 2. Automatically delete the pointed-to object in destructor
    // 3. Support dereference operators (* and ->)
    // 4. Implement move semantics
    // 5. Prevent copying (or implement properly)
};
```

### Exercise 3: Type Traits Detector
Create a function that detects if a type is arithmetic and handles it differently.

```cpp
#include <type_traits>

template<typename T>
void process_value(T value) {
    // TODO: Use SFINAE or if constexpr to:
    // 1. Print "Arithmetic: " followed by the value if T is arithmetic
    // 2. Print "Non-arithmetic: " followed by the type name if T is not arithmetic
}
```

## Solutions to Exercises

### Solution 1: Generic Min/Max Functions
```cpp
#include <type_traits>

template<typename T>
constexpr const T& min(const T& a, const T& b) {
    return (a < b) ? a : b;
}

template<typename T>
constexpr const T& max(const T& a, const T& b) {
    return (a < b) ? b : a;
}

// For primitive types, we might want to return by value
template<typename T>
constexpr std::conditional_t<std::is_arithmetic_v<T>, T, const T&>
min_val(const T& a, const T& b) {
    return (a < b) ? a : b;
}
```

### Solution 2: Smart Pointer Wrapper
```cpp
template<typename T>
class SimplePtr {
private:
    T* ptr;

public:
    explicit SimplePtr(T* p = nullptr) : ptr(p) {}
    
    ~SimplePtr() {
        delete ptr;
    }
    
    // Move constructor
    SimplePtr(SimplePtr&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }
    
    // Move assignment
    SimplePtr& operator=(SimplePtr&& other) noexcept {
        if (this != &other) {
            delete ptr;
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }
    
    // Delete copy operations to prevent double deletion
    SimplePtr(const SimplePtr&) = delete;
    SimplePtr& operator=(const SimplePtr&) = delete;
    
    T& operator*() const { return *ptr; }
    T* operator->() const { return ptr; }
    T* get() const { return ptr; }
};
```

### Solution 3: Type Traits Detector
```cpp
#include <type_traits>
#include <iostream>

template<typename T>
void process_value(T value) {
#if __cplusplus >= 201703L  // C++17 and later
    if constexpr (std::is_arithmetic_v<T>) {
        std::cout << "Arithmetic: " << value << std::endl;
    } else {
        std::cout << "Non-arithmetic: " << typeid(T).name() << std::endl;
    }
#else  // Pre-C++17 approach using SFINAE
    process_value_impl(value, std::is_arithmetic<T>{});
#endif
}

// Pre-C++17 implementation
template<typename T>
void process_value_impl(T value, std::true_type) {
    std::cout << "Arithmetic: " << value << std::endl;
}

template<typename T>
void process_value_impl(T value, std::false_type) {
    std::cout << "Non-arithmetic: " << typeid(T).name() << std::endl;
}
```

## Quiz Questions

1. What is the difference between `auto x = expr;` and `auto& x = expr;`?

2. Explain the difference between lvalue and rvalue references and provide an example of each.

3. What does SFINAE stand for and why is it important in template metaprogramming?

4. What is the purpose of `std::move` and when should it be used?

5. How does RAII help with resource management in C++?

## Summary
Module 1 covered essential modern C++ features that form the foundation for template metaprogramming. Students learned about auto, constexpr, lambdas, RAII, smart pointers, move semantics, basic function templates, type deduction, and SFINAE basics. These concepts are crucial for understanding more advanced template metaprogramming techniques in subsequent modules.