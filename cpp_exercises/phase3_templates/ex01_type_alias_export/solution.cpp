// SOLUTION: ex01_type_alias_export
// Demonstrates type aliases inside class templates

#include <iostream>
#include <type_traits>

// Class template with exported type alias
template<typename T>
class Tensor2D {
public:
    // Export the template parameter as a named type
    // This is the CUTLASS pattern — every policy class does this
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
    
    // Delete copy operations
    Tensor2D(const Tensor2D&) = delete;
    Tensor2D& operator=(const Tensor2D&) = delete;
};

// Function template that queries the exported type
template<typename TensorType>
void print_element_type() {
    // typename is required to access nested dependent types
    // TensorType::value_type depends on the template parameter
    using ElementType = typename TensorType::value_type;
    
    std::cout << "Element type is float: " << std::is_same_v<ElementType, float> << "\n";
    std::cout << "Element type is int: " << std::is_same_v<ElementType, int> << "\n";
}

// Type alias for a specific instantiation
// This is common: create shorthand for commonly used template instantiations
using FloatTensor = Tensor2D<float>;
using IntTensor = Tensor2D<int>;

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
    
    std::cout << "\n=== IntTensor alias ===\n";
    IntTensor it(3, 3);
    std::cout << "IntTensor is Tensor2D<int>: " 
              << std::is_same_v<IntTensor, Tensor2D<int>> << "\n";
    
    std::cout << "\n=== CUTLASS Pattern ===\n";
    std::cout << "CUTLASS uses this pattern everywhere:\n";
    std::cout << "  template<typename T, typename Layout>\n";
    std::cout << "  class TensorRef {\n";
    std::cout << "    using value_type = T;\n";
    std::cout << "    using index_type = int32_t;\n";
    std::cout << "    using Layout = Layout;\n";
    std::cout << "  };\n";
    std::cout << "\nOutside code queries: typename TensorRef<float, Layout>::value_type\n";
    
    return 0;
}

// KEY_INSIGHT:
// Type aliases inside templates export the template parameters as named types.
// Syntax: using alias_name = Type; inside the class template.
//
// Accessing from outside requires 'typename':
//   typename Tensor2D<float>::value_type  // resolves to float
//
// Why 'typename'? Because value_type is a DEPENDENT NAME — it depends on
// the template parameter. The compiler needs 'typename' to know it's a type,
// not a static member or function.
//
// CUTLASS mapping: Every CUTLASS policy class exports types:
//   cutlass::gemm::GemmCoord::IndexType
//   cutlass::TensorRef<T, Layout>::value_type
//   cutlass::arch::Mma<T, Layout, Policy>::FragmentType
// Understanding this pattern is essential for reading CUTLASS source.
