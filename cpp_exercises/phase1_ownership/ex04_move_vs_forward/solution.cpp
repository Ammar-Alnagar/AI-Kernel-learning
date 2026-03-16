// SOLUTION: ex04_move_vs_forward
// Demonstrates the difference between std::move and std::forward

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

// Perfect forwarding wrapper
// Preserves the original value category of the argument
template<typename T>
void wrapper_forward(T&& arg) {
    // std::forward<T>(arg) does:
    // - If T is U& (lvalue ref), returns U& (lvalue)
    // - If T is U&& (rvalue ref), returns U&& (rvalue)
    // This is "reference collapsing" in action
    process(std::forward<T>(arg));
}

// Always-move wrapper
// Always treats the argument as an rvalue
template<typename T>
void wrapper_move(T&& arg) {
    // std::move(arg) always returns an rvalue reference
    // Even if arg was originally an lvalue, it gets cast to rvalue
    // This can cause unintended moves!
    process(std::move(arg));
}

int main() {
    std::string named = "named string";
    
    std::cout << "=== Test 1: wrapper_forward with lvalue ===\n";
    wrapper_forward(named);  // T deduced as std::string&
                             // forward<std::string&> returns lvalue
                             // process(lvalue) called
    std::cout << "named after forward: " << named << " (unchanged)\n";
    
    std::cout << "\n=== Test 2: wrapper_forward with rvalue ===\n";
    wrapper_forward(std::string("temporary"));  // T deduced as std::string
                                                // forward<std::string> returns rvalue
                                                // process(rvalue) called
    
    std::cout << "\n=== Test 3: wrapper_move with lvalue ===\n";
    wrapper_move(named);  // T deduced as std::string&
                          // move casts to rvalue anyway
                          // process(rvalue) called — unintended move!
    std::cout << "named after move: '" << named << "' (empty — was moved from!)\n";
    
    std::cout << "\n=== Test 4: wrapper_move with rvalue ===\n";
    wrapper_move(std::string("another temporary"));  // T deduced as std::string
                                                     // move casts to rvalue
                                                     // process(rvalue) called
    
    std::cout << "\n=== Key Difference ===\n";
    std::cout << "std::forward<T>(x): preserves x's original category\n";
    std::cout << "std::move(x): always casts to rvalue\n";
    std::cout << "\nUse forward in templates that should preserve category.\n";
    std::cout << "Use move when you explicitly want to steal resources.\n";
    
    return 0;
}

// KEY_INSIGHT:
// std::move: Always casts to rvalue. Use when you want to steal.
// std::forward: Preserves original category. Use in templates.
//
// Reference collapsing rules (why forward works):
// T&& with T = U&    → U&  (lvalue ref)
// T&& with T = U&&   → U&& (rvalue ref)
// T&  with T = U     → U&  (lvalue ref)
//
// In wrapper_forward(T&& arg):
// - Called with lvalue: T = U&, so T&& = U& (lvalue ref)
// - Called with rvalue: T = U, so T&& = U&& (rvalue ref)
//
// forward<T>(arg) returns the same category T was deduced as.
//
// CUDA mapping: Perfect forwarding in generic CUDA wrappers:
// template<typename T>
// void launchKernel(T&& arg) {
//     kernel(std::forward<T>(arg));  // Preserves category
// }
// This allows the same wrapper to handle lvalues and rvalues correctly.
