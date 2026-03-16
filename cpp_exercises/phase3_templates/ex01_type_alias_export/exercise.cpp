// CONCEPT: Type aliases inside templates — using value_type = T
// FORMAT: SCAFFOLD
// TIME_TARGET: 15 min
// WHY_THIS_MATTERS: CUTLASS exports types from templates via `using`. You must know this pattern.
// CUDA_CONNECTION: cutlass::TensorRef<T, Layout> exports value_type, index_type, etc.

#include <iostream>
#include <type_traits>

// TODO 1: Add type alias inside the class template
// Inside Tensor2D, add: using value_type = T;
// This exports the template parameter as a named type
template<typename T>
class Tensor2D {
public:
    // TODO: Add type alias here
    // using value_type = T;
    using value_type = T;
    
    T* data;
    size_t rows, cols;
    
    Tensor2D(size_t r, size_t c) 
        : data(new T[r * c]()), rows(r), cols(c) {}
    
    ~Tensor2D() { delete[] data; }
    
    // Move operations
    Tensor2D(Tensor2D&& other) noexcept 
        : data(other.data), rows(other.rows), cols(other.cols) {
        other.data = nullptr;
        other.rows = 0;
        other.cols = 0;
    }
    
    Tensor2D& operator=(Tensor2D&& other) noexcept {
        if (this != &other) {
            delete[] data;
            data = other.data;
            rows = other.rows;
            cols = other.cols;
            other.data = nullptr;
            other.rows = 0;
            other.cols = 0;
        }
        return *this;
    }
    
    // Delete copy operations for simplicity
    Tensor2D(const Tensor2D&) = delete;
    Tensor2D& operator=(const Tensor2D&) = delete;
};

// TODO 2: Write a function that queries the exported type
// Use typename to access the nested type
// Template syntax: typename TensorType::value_type
template<typename TensorType>
void print_element_type() {
    // TODO: Use typename to access the nested type alias
    // std::cout << "Element type: " << typeid(typename TensorType::value_type).name() << "\n";
    using ElementType = typename TensorType::value_type;
    std::cout << "Element type is float: " << std::is_same_v<ElementType, float> << "\n";
    std::cout << "Element type is int: " << std::is_same_v<ElementType, int> << "\n";
}

// TODO 3: Create a type alias for a specific instantiation
// using FloatTensor = Tensor2D<float>;
// This is common in CUTLASS: using TensorRef = cutlass::TensorRef<float, Layout>;
using FloatTensor = Tensor2D<float>;

int main() {
    std::cout << "=== Creating Tensor2D<float> ===\n";
    Tensor2D<float> tensor_f(3, 4);
    std::cout << "Tensor<float> rows=" << tensor_f.rows << ", cols=" << tensor_f.cols << "\n";
    
    std::cout << "\n=== Querying element type ===\n";
    print_element_type<Tensor2D<float>>();
    
    std::cout << "\n=== Creating Tensor2D<int> ===\n";
    Tensor2D<int> tensor_i(2, 2);
    print_element_type<Tensor2D<int>>();
    
    std::cout << "\n=== Using type alias (FloatTensor) ===\n";
    FloatTensor ft(5, 5);
    std::cout << "FloatTensor is Tensor2D<float>: " 
              << std::is_same_v<FloatTensor, Tensor2D<float>> << "\n";
    
    std::cout << "\n=== CUTLASS Pattern ===\n";
    std::cout << "CUTLASS uses this pattern everywhere:\n";
    std::cout << "  template<typename T, typename Layout>\n";
    std::cout << "  class TensorRef {\n";
    std::cout << "    using value_type = T;\n";
    std::cout << "    using index_type = int32_t;\n";
    std::cout << "    using Layout = Layout;\n";
    std::cout << "  };\n";
    
    return 0;
}

// VERIFY: Expected output:
// === Creating Tensor2D<float> ===
// Tensor<float> rows=3, cols=4
//
// === Querying element type ===
// Element type is float: 1
// Element type is int: 0
//
// === Creating Tensor2D<int> ===
// Element type is float: 0
// Element type is int: 1
//
// === Using type alias (FloatTensor) ===
// FloatTensor is Tensor2D<float>: 1
//
// === CUTLASS Pattern ===
// CUTLASS uses this pattern everywhere...
