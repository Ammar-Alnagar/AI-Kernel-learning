// CONCEPT: unique_ptr is move-only, not un-movable
// FORMAT: DEBUG
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: Learner believes unique_ptr cannot be moved. This is wrong.
// CUDA_CONNECTION: unique_ptr<DeviceMemory> transfers ownership between host and kernel launchers.

#include <iostream>
#include <memory>
#include <cstring>

// SYMPTOMS:
// 1. Compile error: "call to deleted constructor" when trying to copy unique_ptr
// 2. Learner thinks: "unique_ptr can't be moved, I need shared_ptr"
// 3. Reality: unique_ptr IS movable, just not copyable

void demonstrate_unique_ptr() {
    std::cout << "=== Creating unique_ptr ===\n";
    std::unique_ptr<int> ptr1 = std::make_unique<int>(42);
    std::cout << "ptr1: " << *ptr1 << " at " << ptr1.get() << "\n";
    
    // BUG: Learner tries to COPY unique_ptr — this fails
    // std::unique_ptr<int> ptr2 = ptr1;  // ERROR: copy constructor deleted
    
    // LEARNER'S WRONG CONCLUSION: "unique_ptr can't be transferred"
    // CORRECT: unique_ptr can be MOVED, just not COPIED
    
    // TODO: Fix by using std::move to transfer ownership
    // std::unique_ptr<int> ptr2 = std::move(ptr1);
    // After this: ptr1 is nullptr, ptr2 owns the int
    
    // For now, this workaround compiles but defeats the purpose:
    std::unique_ptr<int> ptr2 = std::make_unique<int>(*ptr1);  // Deep copy (wrong pattern!)
    std::cout << "ptr2 (copied value): " << *ptr2 << "\n";
    std::cout << "ptr1 still owns original: " << *ptr1 << "\n";
    
    std::cout << "\n=== After scope (both destructors fire) ===\n";
    // Both ptr1 and ptr2 delete their own memory — this is NOT ownership transfer!
}

void demonstrate_correct_move() {
    std::cout << "\n=== Correct: Move unique_ptr ===\n";
    std::unique_ptr<int> ptr1 = std::make_unique<int>(100);
    std::cout << "ptr1 before move: " << ptr1.get() << " (value: " << *ptr1 << ")\n";
    
    // TODO: Transfer ownership using std::move
    // Uncomment and observe:
    // std::unique_ptr<int> ptr2 = std::move(ptr1);
    // std::cout << "ptr1 after move: " << ptr1.get() << " (should be nullptr)\n";
    // std::cout << "ptr2 after move: " << ptr2.get() << " (owns the int)\n";
    
    // Simulating what the output should show:
    std::cout << "(Move not implemented — ptr1 still owns memory)\n";
}

void demonstrate_use_after_move() {
    std::cout << "\n=== Use-after-move (UB!) ===\n";
    std::unique_ptr<int> ptr1 = std::make_unique<int>(200);
    std::unique_ptr<int> ptr2 = std::move(ptr1);
    
    std::cout << "ptr2 owns: " << *ptr2 << "\n";
    
    // BUG: Accessing ptr1 after move — it's nullptr!
    // This is undefined behavior (null pointer dereference)
    // SYMPTOM: Crash or garbage value
    if (ptr1) {
        std::cout << "ptr1 still valid: " << *ptr1 << "\n";  // Won't print — ptr1 is null
    } else {
        std::cout << "ptr1 is nullptr (correct — ownership was transferred)\n";
    }
    
    // TODO: Remove this dangerous line (it's commented out for safety):
    // std::cout << "Dangerous: " << *ptr1 << "\n";  // CRASH: dereferencing nullptr
}

int main() {
    demonstrate_unique_ptr();
    demonstrate_correct_move();
    demonstrate_use_after_move();
    
    std::cout << "\n=== All scopes exited (no leaks!) ===\n";
    return 0;
}

// VERIFY (after fix):
// 1. demonstrate_unique_ptr: ptr1 becomes nullptr after std::move(ptr1)
// 2. demonstrate_correct_move: ptr2 owns the original pointer, ptr1 is null
// 3. demonstrate_use_after_move: ptr1 check prints "nullptr" message, no crash
// 4. No memory leaks — unique_ptr cleans up automatically

// KEY LEARNING:
// unique_ptr IS movable. It is NOT copyable.
// std::move(ptr) transfers ownership, leaves source null.
// shared_ptr is for SHARED ownership — a different concept entirely.
