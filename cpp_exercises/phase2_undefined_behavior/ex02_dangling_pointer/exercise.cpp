// CONCEPT: Dangling pointer — accessing freed memory
// FORMAT: DEBUG
// TIME_TARGET: 10 min
// WHY_THIS_MATTERS: Dangling pointers cause silent corruption or crashes.
// CUDA_CONNECTION: Device memory use-after-free — cudaFree'd pointer accessed in kernel.

#include <iostream>
#include <cstring>

// SYMPTOMS:
// 1. Program may print garbage values
// 2. Program may crash with segfault
// 3. Sanitizer reports: heap-use-after-free
// 4. Behavior differs between runs (UB is non-deterministic)

class DataHolder {
public:
    int* data;
    size_t size;
    
    DataHolder(size_t s) : data(new int[s]), size(s) {
        for (size_t i = 0; i < s; ++i) {
            data[i] = static_cast<int>(i * 10);
        }
    }
    
    ~DataHolder() {
        delete[] data;
    }
    
    void print() const {
        std::cout << "Data: ";
        for (size_t i = 0; i < size; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
    }
};

int* get_dangling_pointer() {
    DataHolder holder(5);
    // BUG: Returning pointer to memory that will be freed when holder goes out of scope
    // This is a DANGLING POINTER — the memory is freed but we return the address
    return holder.data;  // BUG LINE: returning pointer to soon-to-be-freed memory
}

// TODO: Fix by either:
// Option 1: Return the DataHolder itself (RAII — owner manages lifetime)
// Option 2: Allocate data that outlives the function (not recommended for this exercise)
// Option 3: Return a copy of the data (std::vector<int>)

// For this exercise, implement Option 1: return the owner
DataHolder get_holder() {
    // TODO: Create DataHolder(5), fill with values, return by value
    // Move semantics will transfer ownership safely
    DataHolder holder(5);
    for (size_t i = 0; i < 5; ++i) {
        holder.data[i] = static_cast<int>(i * 10);
    }
    return holder;  // Move constructor transfers ownership
}

int main() {
    std::cout << "=== Attempting to get data pointer ===\n";
    
    // BUGGY: This gets a dangling pointer
    // int* ptr = get_dangling_pointer();
    // DataHolder is destroyed, memory freed, ptr is now dangling
    // std::cout << "Accessing dangling pointer: " << ptr[0] << "\n";  // UB!
    
    // FIXED: Get the owner, not the raw pointer
    DataHolder holder = get_holder();
    std::cout << "Got DataHolder, printing contents:\n";
    holder.print();  // Safe — holder owns the memory
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Never return pointers to local data.\n";
    std::cout << "Return the OWNER (RAII object) instead.\n";
    std::cout << "Raw pointers should OBSERVE, not OWN.\n";
    
    return 0;
}

// VERIFY:
// Buggy version: sanitizer reports heap-use-after-free, or garbage output
// Fixed version: prints "Data: 0 10 20 30 40" consistently, no errors

// BUILD COMMAND:
// g++ -std=c++20 -O2 -fsanitize=address -o ex02 exercise.cpp && ./ex02
