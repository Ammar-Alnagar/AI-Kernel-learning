// SOLUTION: ex03_sfinae_vs_concepts
// Demonstrates SFINAE (pre-C++20) vs Concepts (C++20)

#include <iostream>
#include <type_traits>

// ==================== SFINAE (Pre-C++20) ====================

// SFINAE: Substitution Failure Is Not An Error
// When template argument substitution fails, that overload is silently removed
// from the candidate set. If no overload remains, it's a compile error.

// Floating-point overload (SFINAE)
// std::enable_if_t<condition, int> evaluates to int if condition is true
// If condition is false, substitution fails — this overload is removed
template<typename T, std::enable_if_t<std::is_floating_point_v<T>, int> = 0>
void process_sfinae(T value) {
    std::cout << "[SFINAE] Floating-point: " << value << "\n";
}

// Integral overload (SFINAE)
template<typename T, std::enable_if_t<std::is_integral_v<T>, int> = 0>
void process_sfinae(T value) {
    std::cout << "[SFINAE] Integral: " << value << "\n";
}

// ==================== Concepts (C++20) ====================

// Concepts are named constraints on template parameters
// Much cleaner than SFINAE — explicit, readable, better error messages

// Define a concept for floating-point types
template<typename T>
concept FloatingPoint = std::is_floating_point_v<T>;

// Define a concept for integral types
template<typename T>
concept Integral = std::is_integral_v<T>;

// Use concept with 'requires' clause
template<typename T> requires FloatingPoint<T>
void process_concept(T value) {
    std::cout << "[Concept] Floating-point: " << value << "\n";
}

template<typename T> requires Integral<T>
void process_concept(T value) {
    std::cout << "[Concept] Integral: " << value << "\n";
}

// Alternative: abbreviated function template (C++20)
// void process_concept(FloatingPoint auto value);  // Same effect

// ==================== Combined Example ====================

// Concept combining multiple constraints
template<typename T>
concept Numeric = std::is_arithmetic_v<T>;  // integral OR floating-point

template<typename T> requires Numeric<T>
void process_any_numeric(T value) {
    std::cout << "[Concept] Numeric: " << value << "\n";
}

int main() {
    std::cout << "=== SFINAE Version ===\n";
    process_sfinae(3.14);   // Floating-point overload selected
    process_sfinae(42);     // Integral overload selected
    // process_sfinae("hello");  // Compile error: no matching overload
    
    std::cout << "\n=== Concepts Version ===\n";
    process_concept(2.71);  // Floating-point overload selected
    process_concept(100);   // Integral overload selected
    // process_concept("world");  // Compile error: no matching overload
    
    std::cout << "\n=== Any Numeric Type ===\n";
    process_any_numeric(42);     // Works: int is arithmetic
    process_any_numeric(3.14);   // Works: double is arithmetic
    
    std::cout << "\n=== Type Trait Checks ===\n";
    std::cout << "float is floating_point: " << std::is_floating_point_v<float> << "\n";
    std::cout << "int is integral: " << std::is_integral_v<int> << "\n";
    std::cout << "double is integral: " << std::is_integral_v<double> << "\n";
    std::cout << "int is arithmetic: " << std::is_arithmetic_v<int> << "\n";
    
    std::cout << "\n=== CUTLASS Mapping ===\n";
    std::cout << "CUTLASS 2.x uses SFINAE (std::enable_if_t).\n";
    std::cout << "CUTLASS 3.x uses C++20 concepts.\n";
    std::cout << "\nCUTLASS 3.x example:\n";
    std::cout << "  template<typename T>\n";
    std::cout << "  concept is_numeric = /* ... */;\n";
    std::cout << "\n  template<typename T> requires is_numeric<T>\n";
    std::cout << "  class Tensor;\n";
    
    return 0;
}

// KEY_INSIGHT:
// SFINAE (Substitution Failure Is Not An Error):
// - Pre-C++20 technique for constraining templates
// - Uses std::enable_if_t<condition, Type>
// - Verbose, hard to read, confusing error messages
// - Example: template<typename T, std::enable_if_t<is_float<T>, int> = 0>
//
// Concepts (C++20):
// - Named constraints: template<typename T> concept Name = constraint;
// - Clean syntax: template<typename T> requires Name<T>
// - Clear error messages: "constraint not satisfied" vs "substitution failed"
// - Example: template<typename T> requires std::floating_point<T>
//
// CUTLASS mapping: CUTLASS 2.x code is full of enable_if_t. CUTLASS 3.x
// migrated to concepts. You'll encounter both when reading CUTLASS source.
// Concepts make the intent much clearer.
