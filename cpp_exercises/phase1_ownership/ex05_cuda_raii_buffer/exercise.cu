// CONCEPT: RAII wrapper for CUDA device memory
// FORMAT: IMPLEMENT
// TIME_TARGET: 25 min
// WHY_THIS_MATTERS: Manual cudaMalloc/cudaFree is error-prone. RAII ensures cleanup.
// CUDA_CONNECTION: Direct application — device memory wrapper with Rule of Five.

#include <iostream>
#include <stdexcept>
#include <cuda_runtime.h>

// TODO: Implement a RAII wrapper for CUDA device memory
// Requirements:
// 1. Constructor: cudaMalloc, throw std::runtime_error on failure
// 2. Destructor: cudaFree (safe even if allocation failed)
// 3. Copy constructor: DELETED (device memory should not be copied)
// 4. Copy assignment: DELETED
// 5. Move constructor: transfer ownership, nullify source
// 6. Move assignment: free current, transfer ownership, nullify source
// 7. get() method: return raw pointer for kernel launches
// 8. size() method: return allocated size

class DeviceBuffer {
public:
    void* ptr;
    size_t size_bytes;
    
    // Constructor: allocate device memory
    DeviceBuffer(size_t bytes) : ptr(nullptr), size_bytes(bytes) {
        // TODO: cudaMalloc(&ptr, bytes)
        // Throw std::runtime_error if cudaSuccess != result
    }
    
    // Destructor: free device memory
    ~DeviceBuffer() {
        // TODO: if (ptr) cudaFree(ptr)
    }
    
    // TODO: Delete copy operations (device memory is expensive to copy)
    // DeviceBuffer(const DeviceBuffer&) = delete;
    // DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    
    // TODO: Implement move constructor
    // Steal ptr and size, set source.ptr = nullptr, source.size_bytes = 0
    
    // TODO: Implement move assignment
    // Free current ptr, steal source, nullify source
    
    // Accessor: get raw pointer for kernel launch
    void* get() const { return ptr; }
    
    // Accessor: get size
    size_t size() const { return size_bytes; }
};

// CUDA kernel to test the buffer
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
    // Check for CUDA device
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

// VERIFY:
// 1. Program prints allocated address
// 2. Kernels launch successfully
// 3. Host verification shows 6.28 (3.14 * 2.0)
// 4. After move, buf.get() is nullptr, buf2.get() owns memory
// 5. No cudaError, no memory leaks (cuda-memcheck passes)

// BUILD COMMAND:
// nvcc -std=c++17 -O2 -arch=sm_89 -o ex05 exercise.cu && ./ex05
