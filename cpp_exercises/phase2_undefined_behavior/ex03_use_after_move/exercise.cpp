// CONCEPT: Use-after-move — accessing moved-from object
// FORMAT: DEBUG
// TIME_TARGET: 10 min
// WHY_THIS_MATTERS: Moved-from objects are in valid but unspecified state — accessing them is UB.
// CUDA_CONNECTION: Moving a device buffer then launching kernel with old pointer.

#include <iostream>
#include <memory>
#include <cstring>

// SYMPTOMS:
// 1. Program may crash with segfault (null pointer dereference)
// 2. Program may print garbage (accessing freed memory)
// 3. Sanitizer reports: use-after-move or heap-use-after-free
// 4. Behavior differs between runs

class Buffer {
public:
    char* data;
    size_t size;
    
    Buffer(size_t s) : data(new char[s]), size(s) {
        std::memset(data, 0, size);
    }
    
    ~Buffer() {
        delete[] data;
    }
    
    // Move constructor: steal pointer, nullify source
    Buffer(Buffer&& other) noexcept : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    
    // Move assignment
    Buffer& operator=(Buffer&& other) noexcept {
        if (this == &other) return *this;
        delete[] data;
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        return *this;
    }
    
    // Delete copy operations
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    
    void write(const char* msg) {
        std::strncpy(data, msg, size - 1);
    }
    
    void print() const {
        // BUG: No null check — UB if data is nullptr
        std::cout << "Buffer: " << data << "\n";
    }
};

void demonstrate_use_after_move() {
    std::cout << "=== Creating and moving buffer ===\n";
    Buffer buf1(64);
    buf1.write("Original data in buf1");
    
    std::cout << "buf1 before move: ";
    buf1.print();
    
    Buffer buf2 = std::move(buf1);
    std::cout << "buf2 after move: ";
    buf2.print();
    
    // BUG: Accessing buf1 after move — it's in valid but unspecified state
    // In our implementation, buf1.data is nullptr — dereferencing is UB
    std::cout << "buf1 after move: ";
    buf1.print();  // BUG LINE: use-after-move — buf1.data is nullptr!
    
    // TODO: Fix by checking if the object is valid before accessing
    // Option 1: Add isValid() method that checks data != nullptr
    // Option 2: Check before use: if (buf1.data) buf1.print();
    // Option 3: Don't access moved-from objects at all (best practice)
}

void demonstrate_safe_pattern() {
    std::cout << "\n=== Safe pattern: don't access moved-from objects ===\n";
    Buffer buf1(64);
    buf1.write("Data in buf1");
    
    Buffer buf2 = std::move(buf1);
    // buf1 is now moved-from — don't use it!
    // Only use buf2, which now owns the resource
    
    std::cout << "buf2 (owns the data): ";
    buf2.print();
    
    std::cout << "(buf1 is not accessed — safe!)\n";
}

int main() {
    demonstrate_use_after_move();
    demonstrate_safe_pattern();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Moved-from objects are in VALID but UNSPECIFIED state.\n";
    std::cout << "You MAY destroy them or assign to them.\n";
    std::cout << "You MUST NOT read from them (UB).\n";
    std::cout << "\nBest practice: treat moved-from objects as unusable.\n";
    
    return 0;
}

// VERIFY:
// Buggy version: crashes or sanitizer reports use-after-move
// Fixed version: prints buf2's data, never accesses buf1 after move

// BUILD COMMAND:
// g++ -std=c++20 -O2 -fsanitize=address,undefined -o ex03 exercise.cpp && ./ex03
