// SOLUTION: ex05_cuda_raii_buffer
// RAII wrapper for CUDA device memory with Rule of Five

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

class DeviceBuffer {
public:
    void* ptr;
    size_t size_bytes;
    
    // Constructor: allocate device memory
    DeviceBuffer(size_t bytes) : ptr(nullptr), size_bytes(bytes) {
        cudaError_t err = cudaMalloc(&ptr, bytes);
        if (err != cudaSuccess) {
            throw std::runtime_error(cudaGetErrorString(err));
        }
        std::cout << "cudaMalloc succeeded: " << bytes << " bytes\n";
    }
    
    // Destructor: free device memory
    ~DeviceBuffer() {
        if (ptr) {
            cudaFree(ptr);
            std::cout << "cudaFree called\n";
        }
    }
    
    // Delete copy operations — device memory is expensive to copy
    // If you need a copy, make it explicit with a clone() method
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // Move constructor: transfer ownership
    DeviceBuffer(DeviceBuffer&& other) noexcept 
        : ptr(other.ptr), size_bytes(other.size_bytes) {
        other.ptr = nullptr;
        other.size_bytes = 0;
        std::cout << "DeviceBuffer move constructor\n";
    }
    
    // Move assignment: free current, transfer ownership
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) {
            return *this;
        }
        if (ptr) {
            cudaFree(ptr);  // Free current resource
        }
        ptr = other.ptr;
        size_bytes = other.size_bytes;
        other.ptr = nullptr;
        other.size_bytes = 0;
        std::cout << "DeviceBuffer move assignment\n";
        return *this;
    }
    
    // Accessors
    void* get() const { return ptr; }
    size_t size() const { return size_bytes; }
};

// CUDA kernels
__global__ void initKernel(float* data, size_t n, float value) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = value;
    }
}

__global__ void scaleKernel(float* data, size_t n, float factor) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

void test_device_buffer() {
    const size_t N = 1024;
    const size_t bytes = N * sizeof(float);
    
    std::cout << "=== Allocating device buffer ===\n";
    DeviceBuffer buf(bytes);
    std::cout << "Allocated " << bytes << " bytes at " << buf.get() << "\n";
    
    std::cout << "\n=== Launching init kernel ===\n";
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    initKernel<<<blocks, threads>>>(static_cast<float*>(buf.get()), N, 3.14f);
    cudaDeviceSynchronize();
    std::cout << "Initialized device memory with 3.14\n";
    
    std::cout << "\n=== Launching scale kernel ===\n";
    scaleKernel<<<blocks, threads>>>(static_cast<float*>(buf.get()), N, 2.0f);
    cudaDeviceSynchronize();
    std::cout << "Scaled device memory by 2.0 (now 6.28)\n";
    
    std::cout << "\n=== Copying back to host for verification ===\n";
    float* host_data = new float[N];
    cudaMemcpy(host_data, buf.get(), bytes, cudaMemcpyDeviceToHost);
    std::cout << "First 5 values: ";
    for (int i = 0; i < 5; ++i) {
        std::cout << host_data[i] << " ";
    }
    std::cout << "\n";
    delete[] host_data;
    
    std::cout << "\n=== Moving buffer to another DeviceBuffer ===\n";
    DeviceBuffer buf2(std::move(buf));
    std::cout << "buf.get() after move: " << buf.get() << " (should be nullptr)\n";
    std::cout << "buf2.get() after move: " << buf2.get() << " (owns the memory)\n";
    
    std::cout << "\n=== Exiting test (buf2 destructor fires, cudaFree called) ===\n";
}

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0) {
        std::cerr << "No CUDA device found\n";
        return 1;
    }
    
    std::cout << "CUDA device count: " << device_count << "\n";
    test_device_buffer();
    
    std::cout << "\n=== Success! ===\n";
    return 0;
}

// KEY_INSIGHT:
// RAII for CUDA: constructor cudaMalloc, destructor cudaFree.
// Move operations transfer device pointer ownership without copying.
// Copy operations deleted — device memory is too expensive to copy implicitly.
//
// This pattern scales to:
// - DeviceBuffer arrays (unique_ptr<DeviceBuffer[]>)
// - Pinned host memory (cudaMallocHost/cudaFreeHost)
// - CUDA streams (cudaStreamCreate/Destroy)
// - CUDA events (cudaEventCreate/Destroy)
//
// CUTLASS uses this pattern extensively. Every device memory allocation
// is wrapped in an RAII type. This prevents leaks when exceptions occur
// or when early returns happen in complex kernel launchers.
