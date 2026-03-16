// SOLUTION: ex03_use_after_move
// Demonstrates use-after-move UB and safe patterns

#include <iostream>
#include <memory>
#include <cstring>

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
        if (data) {
            std::strncpy(data, msg, size - 1);
        }
    }
    
    // Safe print: checks for null
    void print() const {
        if (data) {
            std::cout << "Buffer: " << data << "\n";
        } else {
            std::cout << "Buffer: (empty/moved-from)\n";
        }
    }
    
    // Query method: check if buffer is valid
    bool isValid() const {
        return data != nullptr;
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
    
    // FIXED: Check validity before accessing moved-from object
    std::cout << "buf1 after move: ";
    if (buf1.isValid()) {
        buf1.print();
    } else {
        std::cout << "(buf1 is moved-from — not accessing)\n";
    }
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

void demonstrate_reuse_after_move() {
    std::cout << "\n=== Reusing moved-from object (safe pattern) ===\n";
    Buffer buf1(64);
    buf1.write("First data");
    
    Buffer buf2 = std::move(buf1);
    // buf1 is moved-from, but we can ASSIGN to it
    buf1 = Buffer(32);  // Re-allocate new buffer
    buf1.write("Reused buf1");
    
    std::cout << "buf1 (reused): ";
    buf1.print();
    std::cout << "buf2 (original owner): ";
    buf2.print();
}

int main() {
    demonstrate_use_after_move();
    demonstrate_safe_pattern();
    demonstrate_reuse_after_move();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Moved-from objects are in VALID but UNSPECIFIED state.\n";
    std::cout << "\nAllowed operations on moved-from object:\n";
    std::cout << "  - Destroy it (destructor)\n";
    std::cout << "  - Assign to it (operator=)\n";
    std::cout << "\nForbidden operations:\n";
    std::cout << "  - Read from it (UB)\n";
    std::cout << "  - Call methods that access moved-from state\n";
    std::cout << "\nBest practice: treat moved-from objects as unusable.\n";
    
    return 0;
}

// KEY_INSIGHT:
// Use-after-move is UB. After std::move(x), x is in valid but unspecified state.
// The standard guarantees:
// - x can be safely destroyed
// - x can be assigned to
// - x's state is unspecified (often nullptr for pointers)
//
// Reading from x after move is UB — the data may be gone, corrupted, or
// belong to another object now.
//
// CUDA mapping: Moving a DeviceBuffer transfers the cudaDevicePtr. Launching
// a kernel with the old (moved-from) pointer uses freed/invalid device memory.
// Always use the new owner after move.
