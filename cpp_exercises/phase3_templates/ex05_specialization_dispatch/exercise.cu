// CONCEPT: Template specialization vs overloading — type dispatch
// FORMAT: SCAFFOLD
// TIME_TARGET: 20 min
// WHY_THIS_MATTERS: CUTLASS dispatches kernels by type (__half vs float vs bfloat16).
// CUDA_CONNECTION: Kernel launcher specialized for __half, float, __nv_bfloat16.

#include <iostream>
#include <type_traits>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

// ==================== Primary Template ====================

// Primary template: generic implementation (fallback)
template<typename T>
struct KernelConfig {
    static constexpr const char* name() { return "generic"; }
    static constexpr int threads_per_block() { return 256; }
};

// ==================== Full Specializations ====================

// TODO 1: Specialize for float
// Syntax: template<> struct KernelConfig<float> { ... };
template<>
struct KernelConfig<float> {
    static constexpr const char* name() { return "float"; }
    static constexpr int threads_per_block() { return 256; }
};

// TODO 2: Specialize for __half (CUDA half precision)
template<>
struct KernelConfig<__half> {
    static constexpr const char* name() { return "__half"; }
    static constexpr int threads_per_block() { return 512; }  // More threads for half
};

// TODO 3: Specialize for __nv_bfloat16
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

// Overloading: different function templates with same name
// This is NOT specialization — it's a different function template
template<typename T>
void process_value(T value) {
    std::cout << "[Generic] Processing: " << value << "\n";
}

// Overload for int (different template, not specialization)
template<typename T>
void process_value(T value, std::enable_if_t<std::is_integral_v<T>, int> = 0) {
    std::cout << "[Integral] Processing: " << value << "\n";
}

// Specialization would be:
// template<>
// void process_value<int>(int value) { ... }

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
    process_value(42);       // Calls integral overload
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
    
    return 0;
}

// VERIFY: Expected output:
// === Template Specialization Dispatch ===
//
// --- Float ---
// Launching kernel for type: float
// Threads per block: 256
// Would launch 4 blocks x 256 threads
//
// --- Half ---
// Launching kernel for type: __half
// Threads per block: 512
// Would launch 2 blocks x 512 threads
//
// --- BFloat16 ---
// Launching kernel for type: __nv_bfloat16
// Threads per block: 512
// Would launch 2 blocks x 512 threads
//
// === Overloading Demo ===
// [Integral] Processing: 42
// [Generic] Processing: 3.14
// [Integral] Processing: 100
//
// === CUTLASS Mapping ===
// CUTLASS uses full specialization for type dispatch...
