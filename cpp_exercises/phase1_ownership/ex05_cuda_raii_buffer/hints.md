# Hints for ex05_cuda_raii_buffer

## H1 — Concept Direction
The pattern is identical to host RAII: constructor allocates, destructor frees. The difference is using `cudaMalloc`/`cudaFree` instead of `new`/`delete`. Check the return value of `cudaMalloc` — it returns `cudaError_t`, not a pointer.

## H2 — Names the Tool
Use `cudaMalloc(&ptr, bytes)` — note the `&ptr`. On error, throw `std::runtime_error(cudaGetErrorString(err))`. In destructor, check `if (ptr)` before `cudaFree(ptr)`.

## H3 — Minimal Usage (Unrelated Context)
```cpp
class CudaStream {
    cudaStream_t stream;
public:
    CudaStream() { cudaStreamCreate(&stream); }
    ~CudaStream() { cudaStreamDestroy(stream); }
    CudaStream(const CudaStream&) = delete;  // Streams shouldn't be copied
    CudaStream(CudaStream&& o) : stream(o.stream) { o.stream = nullptr; }
};
```
