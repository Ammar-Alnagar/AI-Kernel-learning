// CONCEPT: std::move vs std::forward — when to use each
// FORMAT: SCAFFOLD
// TIME_TARGET: 20 min
// WHY_THIS_MATTERS: move always casts to rvalue; forward preserves original category.
// CUDA_CONNECTION: Perfect forwarding in generic CUDA wrapper functions.

#include <iostream>
#include <utility>
#include <string>

// Overloads to demonstrate which is called
void process(const std::string& s) {
    std::cout << "  -> process(lvalue reference): " << s << "\n";
}

void process(std::string&& s) {
    std::cout << "  -> process(rvalue reference): " << s << "\n";
}

// TODO 1: Implement a template that forwards its argument
// Use std::forward<T>(arg) to preserve the original value category
// If called with lvalue, forward produces lvalue → process(lvalue) called
// If called with rvalue, forward produces rvalue → process(rvalue) called
template<typename T>
void wrapper_forward(T&& arg) {
    // TODO: Use std::forward<T>(arg) to call process
    // This is "perfect forwarding" — preserves original category
    process(std::forward<T>(arg));
}

// TODO 2: Implement a template that always moves its argument
// Use std::move(arg) to always cast to rvalue
// Even if called with lvalue, move produces rvalue → process(rvalue) called
template<typename T>
void wrapper_move(T&& arg) {
    // TODO: Use std::move(arg) to call process
    // This always treats arg as an rvalue, even if it was an lvalue
    process(std::move(arg));
}

int main() {
    std::string named = "named string";
    
    std::cout << "=== Test 1: wrapper_forward with lvalue ===\n";
    wrapper_forward(named);  // Should call: process(lvalue reference)
    std::cout << "named after forward: " << named << " (unchanged)\n";
    
    std::cout << "\n=== Test 2: wrapper_forward with rvalue ===\n";
    wrapper_forward(std::string("temporary"));  // Should call: process(rvalue reference)
    
    std::cout << "\n=== Test 3: wrapper_move with lvalue ===\n";
    wrapper_move(named);  // Should call: process(rvalue reference) — forces move!
    std::cout << "named after move: " << named << " (may be empty — was moved from!)\n";
    
    std::cout << "\n=== Test 4: wrapper_move with rvalue ===\n";
    wrapper_move(std::string("another temporary"));  // Should call: process(rvalue reference)
    
    std::cout << "\n=== Key Difference ===\n";
    std::cout << "std::forward<T>(x): preserves x's original category\n";
    std::cout << "std::move(x): always casts to rvalue\n";
    std::cout << "\nUse forward in templates that should preserve category.\n";
    std::cout << "Use move when you explicitly want to steal resources.\n";
    
    return 0;
}

// VERIFY: Expected output:
// === Test 1: wrapper_forward with lvalue ===
//   -> process(lvalue reference): named string
// named after forward: named string (unchanged)
//
// === Test 2: wrapper_forward with rvalue ===
//   -> process(rvalue reference): temporary
//
// === Test 3: wrapper_move with lvalue ===
//   -> process(rvalue reference): named string
// named after move:  (empty — was moved from!)
//
// === Test 4: wrapper_move with rvalue ===
//   -> process(rvalue reference): another temporary
