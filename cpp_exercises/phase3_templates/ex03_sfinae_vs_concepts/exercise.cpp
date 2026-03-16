// CONCEPT: SFINAE vs Concepts — type constraints in templates
// FORMAT: SCAFFOLD
// TIME_TARGET: 20 min
// WHY_THIS_MATTERS: CUTLASS 3.x uses C++20 concepts. Older code uses SFINAE. You must read both.
// CUDA_CONNECTION: Kernel dispatch constrained to floating-point or integer types.

#include <iostream>
#include <type_traits>

// ==================== SFINAE (Pre-C++20) ====================

// SFINAE: Substitution Failure Is Not An Error
// If template substitution fails, the overload is removed (not an error)

// TODO 1: Enable only for floating-point types using std::enable_if_t
// Syntax: template<typename T, std::enable_if_t<condition, int> = 0>
template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
void process_sfinae(T value) {
    std::cout << "[SFINAE] Floating-point: " << value << "\n";
}

// TODO 2: Enable only for integral types
template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void process_sfinae(T value) {
    std::cout << "[SFINAE] Integral: " << value << "\n";
}

// ==================== Concepts (C++20) ====================

// Concepts are cleaner than SFINAE — explicit constraints

// TODO 3: Define a concept for floating-point types
// Syntax: template<typename T> concept name = constraint;
template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

// TODO 4: Define a concept for integral types
template<typename T>
concept Integral = std::is_integral_v<T>;

// TODO 5: Use concept in function template
// Syntax: template<typename T> requires Concept<T>
// Or: template<Concept T> (abbreviated)
template<typename T> requires FloatingPoint<T>
void process_concept(T value) {
    std::cout << "[Concept] Floating-point: " << value << "\n";
}

template<typename T> requires Integral<T>
void process_concept(T value) {
    std::cout << "[Concept] Integral: " << value << "\n";
}

// ==================== Comparison ====================

// SFINAE version (verbose, hard to read):
// template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
// void foo(T x);

// Concepts version (clean, readable):
// template<typename T> requires std::floating_point<T>
// void foo(T x);
// Or even shorter:
// void foo(std::floating_point auto x);

int main() {
    std::cout << "=== SFINAE Version ===\n";
    process_sfinae(3.14);   // Floating-point overload
    process_sfinae(42);     // Integral overload
    // process_sfinae("hello");  // ERROR: no matching overload
    
    std::cout << "\n=== Concepts Version ===\n";
    process_concept(2.71);  // Floating-point overload
    process_concept(100);   // Integral overload
    // process_concept("world");  // ERROR: no matching overload
    
    std::cout << "\n=== Type Trait Checks ===\n";
    std::cout << "float is floating_point: " << std::is_floating_point_v<float> << "\n";
    std::cout << "int is integral: " << std::is_integral_v<int> << "\n";
    std::cout << "double is integral: " << std::is_integral_v<double> << "\n";
    
    std::cout << "\n=== CUTLASS Mapping ===\n";
    std::cout << "CUTLASS 2.x uses SFINAE (std::enable_if_t).\n";
    std::cout << "CUTLASS 3.x uses C++20 concepts.\n";
    std::cout << "Example: template<typename T> requires cutlass::is_numeric<T>\n";
    
    return 0;
}

// VERIFY: Expected output:
// === SFINAE Version ===
// [SFINAE] Floating-point: 3.14
// [SFINAE] Integral: 42
//
// === Concepts Version ===
// [Concept] Floating-point: 2.71
// [Concept] Integral: 100
//
// === Type Trait Checks ===
// float is floating_point: 1
// int is integral: 1
// double is integral: 0
//
// === CUTLASS Mapping ===
// CUTLASS 2.x uses SFINAE...
