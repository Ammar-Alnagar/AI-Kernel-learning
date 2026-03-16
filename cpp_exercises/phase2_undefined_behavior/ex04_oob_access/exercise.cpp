// CONCEPT: Out-of-bounds array access — Undefined Behavior
// FORMAT: DEBUG
// TIME_TARGET: 10 min
// WHY_THIS_MATTERS: OOB access corrupts memory, causes crashes, or leaks data.
// CUDA_CONNECTION: Kernel thread accesses tile beyond shared memory bounds.

#include <iostream>
#include <cstring>

// SYMPTOMS:
// 1. Program may print garbage values
// 2. Program may crash with segfault
// 3. Sanitizer reports: heap-buffer-overflow or stack-buffer-overflow
// 4. May corrupt adjacent variables (silent data corruption)

void demonstrate_oob_access() {
    std::cout << "=== Stack array with OOB access ===\n";
    int arr[5] = {10, 20, 30, 40, 50};
    
    std::cout << "Valid indices: 0 to 4\n";
    std::cout << "Accessing arr[4]: " << arr[4] << " (valid)\n";
    
    // BUG: Accessing index 5 — out of bounds!
    // This reads memory beyond the array — UB
    std::cout << "Accessing arr[5]: " << arr[5] << " (OOB - UB!)\n";  // BUG LINE
    
    // Even worse: writing OOB
    arr[6] = 999;  // BUG LINE: writes to unknown memory
    std::cout << "Wrote 999 to arr[6] (OOB write - UB!)\n";
}

void demonstrate_heap_oob() {
    std::cout << "\n=== Heap array with OOB access ===\n";
    size_t size = 10;
    int* data = new int[size];
    
    for (size_t i = 0; i < size; ++i) {
        data[i] = static_cast<int>(i * 100);
    }
    
    // BUG: Off-by-one error — i <= size instead of i < size
    std::cout << "Printing with off-by-one bug:\n";
    for (size_t i = 0; i <= size; ++i) {  // BUG LINE: should be i < size
        std::cout << "data[" << i << "] = " << data[i] << "\n";
    }
    
    delete[] data;
}

// TODO: Fix both functions
// Fix 1: Remove OOB accesses in demonstrate_oob_access
// Fix 2: Change loop condition to i < size in demonstrate_heap_oob

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

int main() {
    // Uncomment to see UB (with sanitizer):
    // demonstrate_oob_access();
    // demonstrate_heap_oob();
    
    demonstrate_fixed();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Out-of-bounds access is UNDEFINED BEHAVIOR.\n";
    std::cout << "May crash, may corrupt data, may 'work' until it doesn't.\n";
    std::cout << "\nAlways check: index < size before access.\n";
    std::cout << "Prefer std::array or std::vector with .at() (throws on OOB).\n";
    
    return 0;
}

// VERIFY:
// Buggy version: sanitizer reports heap-buffer-overflow or stack-buffer-overflow
// Fixed version: prints valid indices only, no errors

// BUILD COMMAND:
// g++ -std=c++20 -O2 -fsanitize=address -o ex04 exercise.cpp && ./ex04
