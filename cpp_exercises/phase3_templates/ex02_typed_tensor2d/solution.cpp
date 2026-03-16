// SOLUTION: ex02_typed_tensor2d
// Complete implementation of typed tensor with type aliases

#include <iostream>
#include <type_traits>
#include <cstring>

// Tensor2D class template with exported type aliases
template<typename T, typename IndexType = int32_t>
class Tensor2D {
public:
    // Export type aliases — this is the CUTLASS pattern
    using value_type = T;
    using index_type = IndexType;
    using pointer = T*;
    using const_pointer = const T*;
    
private:
    T* data_;
    IndexType rows_;
    IndexType cols_;
    
public:
    // Constructor: allocate and zero-initialize
    Tensor2D(IndexType rows, IndexType cols) 
        : data_(new T[rows * cols]()), rows_(rows), cols_(cols) {
        std::memset(data_, 0, rows * cols * sizeof(T));
    }
    
    // Destructor: free memory
    ~Tensor2D() {
        delete[] data_;
    }
    
    // Move constructor: steal resources
    Tensor2D(Tensor2D&& other) noexcept 
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    // Move assignment: free current, steal, nullify source
    Tensor2D& operator=(Tensor2D&& other) noexcept {
        if (this != &other) {
            delete[] data_;
            data_ = other.data_;
            rows_ = other.rows_;
            cols_ = other.cols_;
            other.data_ = nullptr;
            other.rows_ = 0;
            other.cols_ = 0;
        }
        return *this;
    }
    
    // Delete copy operations (move-only for clarity)
    Tensor2D(const Tensor2D&) = delete;
    Tensor2D& operator=(const Tensor2D&) = delete;
    
    // 2D access operator (non-const)
    T& operator()(IndexType row, IndexType col) {
        return data_[row * cols_ + col];
    }
    
    // 2D access operator (const)
    const T& operator()(IndexType row, IndexType col) const {
        return data_[row * cols_ + col];
    }
    
    // Accessors
    IndexType rows() const { return rows_; }
    IndexType cols() const { return cols_; }
    IndexType size() const { return rows_ * cols_; }
    pointer data() { return data_; }
    const_pointer data() const { return data_; }
};

// Function template that fills tensor with sequential values
// Uses exported type aliases from the tensor
template<typename TensorType>
void fill_sequential(TensorType& tensor) {
    using value_type = typename TensorType::value_type;
    using index_type = typename TensorType::index_type;
    
    for (index_type i = 0; i < tensor.rows(); ++i) {
        for (index_type j = 0; j < tensor.cols(); ++j) {
            tensor(i, j) = static_cast<value_type>(i * tensor.cols() + j);
        }
    }
}

// Function template that prints tensor contents
template<typename TensorType>
void print_tensor(const TensorType& tensor, const char* label) {
    std::cout << label << " (" << tensor.rows() << "x" << tensor.cols() << "):\n";
    for (typename TensorType::index_type i = 0; i < tensor.rows(); ++i) {
        std::cout << "  ";
        for (typename TensorType::index_type j = 0; j < tensor.cols(); ++j) {
            std::cout << tensor(i, j) << " ";
        }
        std::cout << "\n";
    }
}

int main() {
    std::cout << "=== Creating Tensor2D<float, int32_t> ===\n";
    Tensor2D<float> tensor_f(3, 4);
    
    std::cout << "\n=== Filling with sequential values ===\n";
    fill_sequential(tensor_f);
    print_tensor(tensor_f, "tensor_f");
    
    std::cout << "\n=== Creating Tensor2D<int, size_t> ===\n";
    Tensor2D<int, size_t> tensor_i(2, 5);
    fill_sequential(tensor_i);
    print_tensor(tensor_i, "tensor_i");
    
    std::cout << "\n=== Type alias verification ===\n";
    std::cout << "Tensor2D<float>::value_type is float: " 
              << std::is_same_v<Tensor2D<float>::value_type, float> << "\n";
    std::cout << "Tensor2D<int, size_t>::index_type is size_t: " 
              << std::is_same_v<Tensor2D<int, size_t>::index_type, size_t> << "\n";
    
    std::cout << "\n=== Move semantics test ===\n";
    Tensor2D<float> tensor_move(2, 2);
    fill_sequential(tensor_move);
    std::cout << "Before move:\n";
    print_tensor(tensor_move, "tensor_move");
    
    Tensor2D<float> tensor_moved = std::move(tensor_move);
    std::cout << "After move:\n";
    std::cout << "tensor_move.data() = " << tensor_move.data() << " (should be nullptr)\n";
    print_tensor(tensor_moved, "tensor_moved");
    
    return 0;
}

// KEY_INSIGHT:
// Type aliases in templates enable generic code that queries tensor properties.
// 
// Pattern:
//   template<typename T, typename IndexType = int32_t>
//   class Tensor {
//     using value_type = T;
//     using index_type = IndexType;
//   };
//
// Usage in generic code:
//   template<typename TensorType>
//   void process(TensorType& t) {
//     using value_type = typename TensorType::value_type;
//     // Now we can use value_type regardless of what T was
//   }
//
// CUTLASS mapping: This is exactly how cutlass::TensorRef works.
// Every CUTLASS algorithm is templated on tensor types and queries
// value_type, index_type, Layout, etc. from the tensor. This enables
// a single GEMM implementation to work with float, half, bfloat16, etc.
