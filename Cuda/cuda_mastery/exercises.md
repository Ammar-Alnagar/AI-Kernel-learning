# CUDA Mastery Exercises

Progressive exercises to test and deepen your CUDA understanding.

---

## Level 1: Basics

### Exercise 1.1: Vector Operations
Implement the following kernels:
1. Vector subtraction: `C[i] = A[i] - B[i]`
2. Element-wise multiplication: `C[i] = A[i] * B[i]`
3. Vector scaling: `C[i] = alpha * A[i]`

### Exercise 1.2: Indexing Challenge
Write a kernel that processes a 2D image (width × height):
- Use 2D grid and 2D blocks
- Each thread sets pixel value to: `pixel[x,y] = x + y`
- Verify the output matches expected pattern

### Exercise 1.3: Grid-Stride Loop
Modify the vector addition to use grid-stride loop:
- Handle arrays larger than total thread count
- Test with array size = 10× total threads

---

## Level 2: Memory

### Exercise 2.1: Memory Coalescing
Compare coalesced vs uncoalesced access:
1. Create kernel with strided access (stride = 32)
2. Time both versions with cudaEvent
3. Calculate bandwidth for each
4. Report speedup ratio

### Exercise 2.2: SoA vs AoS
Given `struct Point3D { float x, y, z; }`:
1. Implement AoS version that processes only x-coordinates
2. Implement SoA version with separate x[], y[], z[] arrays
3. Benchmark and compare performance

### Exercise 2.3: Constant Memory
Use constant memory for a convolution kernel:
1. Store 9-element convolution kernel in constant memory
2. Apply 1D convolution to input array
3. Compare with global memory version

---

## Level 3: Shared Memory

### Exercise 3.1: Block Reduction
Implement parallel sum reduction:
1. Load data to shared memory
2. Perform tree-based reduction
3. Write block sum to output
4. Extend to full array reduction (multi-kernel or atomics)

### Exercise 3.2: Histogram
Create a histogram using shared memory:
1. Input: array of integers (0-255 range)
2. Use shared memory for bin counts
3. Handle atomic conflicts
4. Output: 256-bin histogram

### Exercise 3.3: Matrix Transpose
Optimize matrix transpose:
1. Implement naive version (uncoalesced writes)
2. Implement shared memory tiled version
3. Compare performance
4. Try different tile sizes (8, 16, 32)

---

## Level 4: Synchronization

### Exercise 4.1: Prefix Sum (Scan)
Implement parallel prefix sum:
1. Use shared memory
2. Koggin-Stone or Blelloch algorithm
3. Handle large arrays (multiple blocks)
4. Verify correctness

### Exercise 4.2: Lock Implementation
Build a simple spinlock:
1. Use atomicCAS to acquire lock
2. Critical section increments counter
3. atomicExch to release
4. Measure contention with many threads

### Exercise 4.3: Warp-Level Reduction
Optimize reduction using warp shuffles:
1. Shared memory reduction to 32 elements
2. Warp shuffle for final reduction
3. Compare with pure shared memory version

---

## Level 5: Optimization

### Exercise 5.1: Occupancy Tuning
Find optimal configuration:
1. Use `cudaOccupancyMaxPotentialBlockSize()`
2. Try `-maxrregcount` flag
3. Measure performance at different occupancies
4. Plot occupancy vs performance

### Exercise 5.2: Register Analysis
Analyze register usage:
1. Compile with `-ptxas-options=-v`
2. Identify register-heavy kernels
3. Refactor to reduce register pressure
4. Measure occupancy improvement

### Exercise 5.3: Memory Bandwidth
Measure achieved bandwidth:
1. Simple copy kernel
2. Calculate: `bytes / time`
3. Compare with peak bandwidth
4. Optimize to reach >80% of peak

---

## Level 6: Advanced

### Exercise 6.1: Stream Concurrency
Implement concurrent execution:
1. Create 4 streams
2. Split work across streams
3. Overlap H2D, kernel, D2H
4. Visualize with Nsight Systems

### Exercise 6.2: Unified Memory
Migrate code to unified memory:
1. Replace cudaMalloc with cudaMallocManaged
2. Remove explicit cudaMemcpy
3. Add prefetching hints
4. Compare performance

### Exercise 6.3: Multi-GPU
Extend to multiple GPUs:
1. Query available devices
2. Split data across GPUs
3. Use separate streams per GPU
4. Aggregate results

---

## Challenge Problems

### Challenge 1: Convolutional Layer
Implement a 2D convolution for neural networks:
- Input: N×H×W×C (batch, height, width, channels)
- Kernel: K×K×C_in×C_out
- Use shared memory tiling
- Optimize for your GPU

### Challenge 2: Matrix Multiplication with Tensor Cores
Implement GEMM using WMMA (Warp Matrix Multiply Accumulate):
- Use tensor cores (Volta+)
- FP16 computation
- Compare with FP32 tiled version

### Challenge 3: Sort Algorithm
Implement parallel sort:
- Bitonic sort or radix sort
- Use shared memory
- Handle arbitrary input sizes

---

## Verification Tips

```cpp
// Verify kernel results
bool verify(float *gpu, float *cpu, int n, float tolerance = 1e-5) {
    for (int i = 0; i < n; i++) {
        float diff = fabs(gpu[i] - cpu[i]);
        if (diff > tolerance) {
            printf("Mismatch at %d: GPU=%.4f CPU=%.4f\n", i, gpu[i], cpu[i]);
            return false;
        }
    }
    return true;
}

// Timing template
float timeKernel(cudaStream_t stream = 0) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, stream);
    
    kernel<<<grid, block, 0, stream>>>(...);
    
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return ms;
}
```

---

## Profiling Tools

| Tool | Purpose |
|------|---------|
| `nvprof` | Command-line profiler |
| Nsight Systems | Timeline visualization |
| Nsight Compute | Kernel-level analysis |
| `cuda-gdb` | Debugging |
| `cuda-memcheck` | Memory error detection |

### Example nvprof usage:
```bash
nvprof ./program
nvprof --metrics all ./program
nvprof --events all ./program
```

### Example Nsight Compute:
```bash
ncu --set full ./program
```
