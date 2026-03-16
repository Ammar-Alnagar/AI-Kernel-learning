// SOLUTION: ex03_uniqueptr_move_only
// Demonstrates that unique_ptr IS movable (just not copyable)

#include <iostream>
#include <memory>
#include <cstring>

void demonstrate_unique_ptr() {
    std::cout << "=== Creating unique_ptr ===\n";
    std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
    std::cout << "ptr1: " << *ptr1 << " at " << ptr1.get() << "\n";
    
    // FIXED: Use std::move to transfer ownership
    // unique_ptr is MOVE-ONLY: can be moved, cannot be copied
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    
    // After move: ptr1 is nullptr, ptr2 owns the int
    std::cout << "ptr1 after move: " << ptr1.get() << " (nullptr — ownership transferred)\n";
    std::cout << "ptr2 after move: " << *ptr2 << " at " << ptr2.get() << " (owns the int)\n";
    
    std::cout << "\n=== After scope (only ptr2's destructor fires) ===\n";
    // Only ptr2 deletes the int — ptr1 is nullptr, delete nullptr is no-op
}

void demonstrate_correct_move() {
    std::cout << "\n=== Correct: Move unique_ptr ===\n";
    std::unique_ptr<int> ptr1 = std::make_unique<int>(100);
    std::cout << "ptr1 before move: " << ptr1.get() << " (value: " << *ptr1 << ")\n";
    
    // Transfer ownership using std::move
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    
    std::cout << "ptr1 after move: " << ptr1.get() << " (nullptr)\n";
    std::cout << "ptr2 after move: " << ptr2.get() << " (value: " << *ptr2 << ")\n";
    
    // ptr1 is now nullptr — safe to check, safe to destroy
    // ptr2 now owns the original memory
}

void demonstrate_use_after_move() {
    std::cout << "\n=== Use-after-move (checking for nullptr) ===\n";
    std::unique_ptr<int> ptr1 = std::make_unique<int>(200);
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    
    std::cout << "ptr2 owns: " << *ptr2 << "\n";
    
    // CORRECT: Check if ptr1 is null before dereferencing
    // After move, ptr1 is guaranteed to be nullptr
    if (ptr1) {
        std::cout << "ptr1 still valid: " << *ptr1 << "\n";
    } else {
        std::cout << "ptr1 is nullptr (correct — ownership was transferred)\n";
    }
    
    // DO NOT DO THIS:
    // std::cout << *ptr1 << "\n";  // UB: dereferencing nullptr
}

void demonstrate_unique_ptr_array() {
    std::cout << "\n=== unique_ptr with arrays ===\n";
    // unique_ptr can manage arrays too
    std::unique_ptr<int[]> arr = std::make_unique<int[]>(5);
    
    for (int i = 0; i < 5; ++i) {
        arr[i] = i * 10;
    }
    
    std::cout << "Array contents: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << arr[i] << " ";
    }
    std::cout << "\n";
    
    // Can move unique_ptr arrays too
    std::unique_ptr<int[]> arr2 = std::move(arr);
    std::cout << "arr after move: " << (arr ? "valid" : "nullptr") << "\n";
    std::cout << "arr2 after move: " << arr2[2] << " (owns the array)\n";
}

int main() {
    demonstrate_unique_ptr();
    demonstrate_correct_move();
    demonstrate_use_after_move();
    demonstrate_unique_ptr_array();
    
    std::cout << "\n=== All scopes exited (no leaks!) ===\n";
    return 0;
}

// KEY_INSIGHT:
// unique_ptr is MOVE-ONLY, not un-movable.
//
// Copy: unique_ptr<int> p2 = p1;           // ERROR: deleted
// Move: unique_ptr<int> p2 = std::move(p1); // OK: transfers ownership
//
// After std::move(p1):
// - p1 is nullptr (guaranteed by standard)
// - p2 owns the original resource
// - Only p2's destructor will delete the resource
//
// shared_ptr is for SHARED ownership (reference counting).
// unique_ptr is for EXCLUSIVE ownership (zero overhead).
//
// CUDA mapping: unique_ptr<DeviceMemory, Deleter> is the standard pattern
// for RAII device memory. Moving a unique_ptr transfers device ownership
// from host builder to kernel launcher. No reference counting overhead.
