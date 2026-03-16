// SOLUTION: ex05_specialization_dispatch
// Demonstrates template specialization for CUDA type dispatch

#include <iostream>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ==================== Primary Template ====================

// Primary template: generic fallback implementation
template<typename T>
struct KernelConfig {
    static constexpr const char* name() { return "generic"; }
    static constexpr int threads_per_block() { return 256; }
};

// ==================== Full Specializations ====================

// Specialization for float
template<>
struct KernelConfig<float> {
    static constexpr const char* name() { return "float"; }
    static constexpr int threads_per_block() { return 256; }
};

// Specialization for __half (CUDA half precision)
// Half precision allows more threads per block (less register pressure)
template<>
struct KernelConfig<__half> {
    static constexpr const char* name() { return "__half"; }
    static constexpr int threads_per_block() { return 512; }
};

// Specialization for __nv_bfloat16
template<>
struct KernelConfig<__nv_bfloat16> {
    static constexpr const char* name() { return "__nv_bfloat16"; }
    static constexpr int threads_per_block() { return 512; }
};

// ==================== Kernel Launch ====================

// CUDA kernel (dummy for demonstration)
template<typename T>
__global__ void dummyKernel(T* data, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + T(1);
    }
}

// Type dispatch function
// Uses specialized KernelConfig<T> for each type
template<typename T>
void launch_kernel(T* d_data, size_t n) {
    std::cout << "Launching kernel for type: " << KernelConfig<T>::name() << "\n";
    std::cout << "Threads per block: " << KernelConfig<T>::threads_per_block() << "\n";
    
    int threads = KernelConfig<T>::threads_per_block();
    int blocks = (n + threads - 1) / threads;
    
    // In real code: dummyKernel<T><<<blocks, threads>>>(d_data, n);
    std::cout << "Would launch " << blocks << " blocks x " << threads << " threads\n";
}

// ==================== Overloading vs Specialization ====================

// Generic function template
template<typename T>
void process_value(T value) {
    std::cout << "[Generic] Processing: " << value << "\n";
}

// Overload for integral types (uses SFINAE to constrain)
// This is OVERLOADING, not specialization — different function template
template<typename T>
void process_value(T value, std::enable_if_t<std::is_integral_v<T>, int> = 0) {
    std::cout << "[Integral] Processing: " << value << "\n";
}

// Full specialization example (for comparison)
// This is SPECIALIZATION — same function template, specific type
template<>
void process_value<int>(int value) {
    std::cout << "[Specialized int] Processing: " << value << "\n";
}

int main() {
    std::cout << "=== Template Specialization Dispatch ===\n";
    
    // Allocate dummy device memory
    float* d_float;
    __half* d_half;
    __nv_bfloat16* d_bf16;
    
    cudaMalloc(&d_float, 1024 * sizeof(float));
    cudaMalloc(&d_half, 1024 * sizeof(__half));
    cudaMalloc(&d_bf16, 1024 * sizeof(__nv_bfloat16));
    
    std::cout << "\n--- Float ---\n";
    launch_kernel(d_float, 1024);
    
    std::cout << "\n--- Half ---\n";
    launch_kernel(d_half, 1024);
    
    std::cout << "\n--- BFloat16 ---\n";
    launch_kernel(d_bf16, 1024);
    
    std::cout << "\n=== Overloading Demo ===\n";
    process_value(42);       // Calls specialized int version
    process_value(3.14);     // Calls generic (no matching overload)
    process_value(100L);     // Calls integral overload (long is integral)
    
    // Cleanup
    cudaFree(d_float);
    cudaFree(d_half);
    cudaFree(d_bf16);
    
    std::cout << "\n=== CUTLASS Mapping ===\n";
    std::cout << "CUTLASS uses full specialization for type dispatch:\n";
    std::cout << "  template<> struct Mma<float, ...> { /* FP32 kernel */ };\n";
    std::cout << "  template<> struct Mma<__half, ...> { /* FP16 kernel */ };\n";
    std::cout << "  template<> struct Mma<__nv_bfloat16, ...> { /* BF16 kernel */ };\n";
    std::cout << "\nOverloading vs Specialization:\n";
    std::cout << "  Overloading: Different function templates, selected by overload resolution\n";
    std::cout << "  Specialization: Same template, specific type gets custom implementation\n";
    
    return 0;
}

// KEY_INSIGHT:
// Full specialization: template<> struct Name<Type> { ... };
// - Provides custom implementation for a specific type
// - Primary template is the fallback
// - CUTLASS uses this for type-specific kernel implementations
//
// Overloading: Different function templates with same name
// - Selected by overload resolution rules
// - Can use SFINAE/concepts to constrain which types match
// - More flexible but can be ambiguous
//
// CUTLASS mapping: Every GEMM variant is a full specialization:
//   cutlass::gemm::device::Gemm<float, ...>
//   cutlass::gemm::device::Gemm<__half, ...>
//   cutlass::gemm::device::Gemm<__nv_bfloat16, ...>
// Each specialization has different kernel implementations optimized
// for that specific data type's characteristics.
