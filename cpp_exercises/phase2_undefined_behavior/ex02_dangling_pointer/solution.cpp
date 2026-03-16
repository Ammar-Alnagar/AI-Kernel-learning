// SOLUTION: ex02_dangling_pointer
// Demonstrates dangling pointer UB and RAII fix

#include <iostream>
#include <cstring>

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
    
    // Move constructor for safe transfer of ownership
    DataHolder(DataHolder&& other) noexcept 
        : data(other.data), size(other.size) {
        other.data = nullptr;
        other.size = 0;
    }
    
    // Move assignment
    DataHolder& operator=(DataHolder&& other) noexcept {
        if (this == &other) return *this;
        delete[] data;
        data = other.data;
        size = other.size;
        other.data = nullptr;
        other.size = 0;
        return *this;
    }
    
    // Delete copy operations (move-only for clarity)
    DataHolder(const DataHolder&) = delete;
    DataHolder& operator=(const DataHolder&) = delete;
    
    void print() const {
        std::cout << "Data: ";
        for (size_t i = 0; i < size; ++i) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";
    }
};

// BUGGY VERSION (for reference):
// int* get_dangling_pointer() {
//     DataHolder holder(5);
//     return holder.data;  // UB: holder destroyed, pointer dangles
// }

// FIXED VERSION: Return the owner, not the raw pointer
DataHolder get_holder() {
    DataHolder holder(5);
    for (size_t i = 0; i < 5; ++i) {
        holder.data[i] = static_cast<int>(i * 10);
    }
    return holder;  // Move constructor transfers ownership safely
}

int main() {
    std::cout << "=== Getting DataHolder (RAII owner) ===\n";
    
    // FIXED: Get the owner, not a raw pointer
    DataHolder holder = get_holder();
    std::cout << "Got DataHolder, printing contents:\n";
    holder.print();  // Safe — holder owns the memory
    
    std::cout << "\n=== Modifying data through owner ===\n";
    holder.data[0] = 999;
    std::cout << "Modified first element to 999:\n";
    holder.print();
    
    std::cout << "\n=== KEY LEARNING ===\n";
    std::cout << "Never return pointers to local data.\n";
    std::cout << "Return the OWNER (RAII object) instead.\n";
    std::cout << "Raw pointers should OBSERVE, not OWN.\n";
    std::cout << "\nRule: The function that allocates should deallocate,\n";
    std::cout << "OR transfer ownership to the caller via RAII.\n";
    
    return 0;
}

// KEY_INSIGHT:
// Dangling pointer: pointer to memory that has been freed.
// Accessing it is UB — may crash, may return garbage, may "work" until it doesn't.
//
// Fix: Return the RAII owner, not the raw pointer.
// The owner's destructor frees the memory when the owner goes out of scope.
// Move semantics transfer ownership safely.
//
// Alternative: Return std::unique_ptr<T> — explicit ownership transfer.
// Alternative: Return std::vector<T> — copy/move the data itself.
//
// CUDA mapping: Returning a device pointer from a host function that
// allocates with cudaMalloc is safe IF the caller is responsible for
// cudaFree. Better: return a DeviceBuffer RAII wrapper that auto-frees.
