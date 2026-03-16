// SOLUTION: ex04_oob_access
// Demonstrates out-of-bounds access UB and safe patterns

#include <iostream>
#include <cstring>
#include <array>
#include <vector>

void demonstrate_oob_access() {
    std::cout << "=== Stack array with OOB access (BUGGY) ===\n";
    int arr[5] = {10, 20, 30, 40, 50};
    
    std::cout << "Valid indices: 0 to 4\n";
    std::cout << "Accessing arr[4]: " << arr[4] << " (valid)\n";
    
    // BUGGY: Accessing index 5 — out of bounds!
    // std::cout << "Accessing arr[5]: " << arr[5] << " (OOB - UB!)\n";
    // arr[6] = 999;  // OOB write
    
    std::cout << "(OOB accesses removed — would trigger sanitizer)\n";
}

void demonstrate_heap_oob() {
    std::cout << "\n=== Heap array with OOB access (BUGGY) ===\n";
    size_t size = 10;
    int* data = new int[size];
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<int>(i * 100);
    }
    
    // FIXED: Correct loop condition
    std::cout << "Printing with correct bounds:\n";
    for (size_t i = 0; i < size; ++i) {  // Fixed: i < size
        std::cout << "data[" << i << "] = " << data[i] << "\n";
    }
    
    delete[] data;
}

void demonstrate_fixed() {
    std::cout << "\n=== Fixed: bounds-checked access ===\n";
    int arr[5] = {10, 20, 30, 40, 50};
    
    // Safe: iterate within bounds
    for (size_t i = 0; i < 5; ++i) {
        std::cout << "arr[" << i << "] = " << arr[i] << "\n";
    }
    
    // Safe: check bounds before access
    size_t idx = 7;
    if (idx < 5) {
        std::cout << "arr[" << idx << "] = " << arr[idx] << "\n";
    } else {
        std::cout << "Index " << idx << " out of bounds (0-4)\n";
    }
}

void demonstrate_safe_containers() {
    std::cout << "\n=== Safe: std::array with .at() ===\n";
    std::array<int, 5> arr = {10, 20, 30, 40, 50};
    
    // .at() throws std::out_of_range on OOB
    try {
        std::cout << "arr.at(4) = " << arr.at(4) << "\n";
        std::cout << "arr.at(5) = " << arr.at(5) << "\n";  // Throws!
    } catch (const std::out_of_range& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }
    
    std::cout << "\n=== Safe: std::vector with .at() ===\n";
    std::vector<int> vec = {100, 200, 300};
    
    try {
        std::cout << "vec.at(2) = " << vec.at(2) << "\n";
        std::cout << "vec.at(3) = " << vec.at(3) << "\n";  // Throws!
    } catch (const std::out_of_range& e) {
        std::cout << "Caught: " << e.what() << "\n";
    }
}

int main() {
    demonstrate_oob_access();
    demonstrate_heap_oob();
    demonstrate_fixed();
    demonstrate_safe_containers();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Out-of-bounds access is UNDEFINED BEHAVIOR.\n";
    std::cout << "May crash, may corrupt data, may 'work' until it doesn't.\n";
    std::cout << "\nSafe patterns:\n";
    std::cout << "  1. Check: if (idx < size) before access\n";
    std::cout << "  2. Use std::array or std::vector with .at()\n";
    std::cout << "  3. Use range-based for loops: for (auto& x : container)\n";
    std::cout << "  4. Use iterators with proper end checks\n";
    
    return 0;
}

// KEY_INSIGHT:
// OOB access is UB — the compiler assumes indices are valid.
// Consequences:
// - Read OOB: garbage values, adjacent variable corruption
// - Write OOB: corrupts adjacent memory, security vulnerabilities
// - May work in Debug, crash in Release (different memory layout)
//
// Safe alternatives:
// - std::array::at() / std::vector::at() — throw on OOB
// - Range-based for loops — no index arithmetic
// - Bounds checking in debug builds (assert, std::span with checks)
//
// CUDA mapping: Kernel threads must check bounds before accessing arrays.
// CUTLASS uses `if (threadIdx.x < TILE_SIZE)` guards everywhere. OOB
// access in shared memory corrupts neighboring thread data.
