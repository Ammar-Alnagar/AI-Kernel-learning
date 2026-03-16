// CONCEPT: Implement a typed tensor with type aliases
// FORMAT: IMPLEMENT
// TIME_TARGET: 25 min
// WHY_THIS_MATTERS: Building fluency with template type exports.
// CUDA_CONNECTION: CUTLASS TensorRef pattern — the foundation of all tensor operations.

#include <iostream>
#include <type_traits>
#include <cstring>

// TODO: Implement a Tensor2D class template with these requirements:
// 1. Template parameters: typename T (element type), typename IndexType = int32_t
// 2. Export type aliases: value_type, index_type, pointer, const_pointer
// 3. Constructor: allocate T[rows * cols], store dimensions
// 4. Destructor: delete[]
// 5. Operator(): 2D access data[row * cols + col]
// 6. Move constructor and move assignment
// 7. Delete copy operations
// 8. Methods: rows(), cols(), size(), data()

template<typename T, typename IndexType = int32_t>
class Tensor2D {
public:
    // TODO: Export type aliases
    using value_type = T;
    using index_type = IndexType;
    using pointer = T*;
    using const_pointer = const T*;
    
private:
    T* data_;
    IndexType rows_;
    IndexType cols_;
    
public:
    // Constructor
    Tensor2D(IndexType rows, IndexType cols) 
        : data_(new T[rows * cols]()), rows_(rows), cols_(cols) {
        std::memset(data_, 0, rows * cols * sizeof(T));
    }
    
    // Destructor
    ~Tensor2D() {
        delete[] data_;
    }
    
    // Move constructor
    Tensor2D(Tensor2D&& other) noexcept 
        : data_(other.data_), rows_(other.rows_), cols_(other.cols_) {
        other.data_ = nullptr;
        other.rows_ = 0;
        other.cols_ = 0;
    }
    
    // Move assignment
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
    
    // Delete copy operations
    Tensor2D(const Tensor2D&) = delete;
    Tensor2D& operator=(const Tensor2D&) = delete;
    
    // 2D access operator
    T& operator()(IndexType row, IndexType col) {
        return data_[row * cols_ + col];
    }
    
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

// TODO: Implement a function template that works with any Tensor2D
// Requirements:
// 1. Template on the tensor type (not element type directly)
// 2. Use typename TensorType::value_type to get element type
// 3. Fill tensor with sequential values: 0, 1, 2, ...
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

// TODO: Implement a function that prints tensor contents
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

// VERIFY: Expected output:
// === Creating Tensor2D<float, int32_t> ===
//
// === Filling with sequential values ===
// tensor_f (3x4):
//   0 1 2 3 
//   4 5 6 7 
//   8 9 10 11 
//
// === Creating Tensor2D<int, size_t> ===
// tensor_i (2x5):
//   0 1 2 3 4 
//   5 6 7 8 9 
//
// === Type alias verification ===
// Tensor2D<float>::value_type is float: 1
// Tensor2D<int, size_t>::index_type is size_t: 1
//
// === Move semantics test ===
// Before move:
// tensor_move (2x2):
//   0 1 
//   2 3 
// After move:
// tensor_move.data() = 0 (nullptr)
// tensor_moved (2x2):
//   0 1 
//   2 3 
