// ============================================================================
// CUDA Syntax Quick Reference
// ============================================================================
// A comprehensive cheat sheet for CUDA programming
// ============================================================================

/* ============================================================================
   1. KERNEL DEFINITION AND LAUNCH
   ============================================================================ */

// Kernel definition (runs on GPU, called from CPU)
__global__ void myKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

// Kernel launch
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
myKernel<<<blocksPerGrid, threadsPerBlock>>>(d_data, n);

// 2D/3D launch configuration
dim3 blockSize(16, 16);           // 2D block
dim3 gridSize((width + 15) / 16, (height + 15) / 16);  // 2D grid
myKernel<<<gridSize, blockSize>>>(...);

dim3 blockSize3D(8, 8, 8);        // 3D block
dim3 gridSize3D(4, 4, 4);         // 3D grid

// Launch with stream and shared memory
size_t sharedMemBytes = 256 * sizeof(float);
cudaStream_t stream;
cudaStreamCreate(&stream);
myKernel<<<gridSize, blockSize, sharedMemBytes, stream>>>(...);


/* ============================================================================
   2. THREAD INDEXING
   ============================================================================ */

// 1D indexing
int tid = threadIdx.x;           // Thread index in block (0 to blockDim.x-1)
int bid = blockIdx.x;            // Block index in grid
int globalId = bid * blockDim.x + tid;

// 2D indexing
int tx = threadIdx.x, ty = threadIdx.y;
int bx = blockIdx.x, by = blockIdx.y;
int globalX = bx * blockDim.x + tx;
int globalY = by * blockDim.y + ty;
int width = gridDim.x * blockDim.x;
int globalId2D = globalY * width + globalX;  // Row-major

// 3D indexing
int tx = threadIdx.x, ty = threadIdx.y, tz = threadIdx.z;
int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;
int x = bx * blockDim.x + tx;
int y = by * blockDim.y + ty;
int z = bz * blockDim.z + tz;
int volumeSize = gridDim.x * blockDim.x * gridDim.y * blockDim.y * gridDim.z * blockDim.z;


/* ============================================================================
   3. BUILT-IN VARIABLES
   ============================================================================ */

threadIdx.x, threadIdx.y, threadIdx.z   // Thread coordinates in block
blockIdx.x, blockIdx.y, blockIdx.z      // Block coordinates in grid
blockDim.x, blockDim.y, blockDim.z      // Block dimensions (threads per block)
gridDim.x, gridDim.y, gridDim.z         // Grid dimensions (blocks per grid)
warpSize                                // Number of threads in warp (always 32)


/* ============================================================================
   4. MEMORY MANAGEMENT
   ============================================================================ */

// Device memory allocation
float *d_ptr;
cudaMalloc(&d_ptr, size_in_bytes);
cudaFree(d_ptr);

// Memory transfers
cudaMemcpy(d_dest, h_src, size, cudaMemcpyHostToDevice);      // H2D
cudaMemcpy(h_dest, d_src, size, cudaMemcpyDeviceToHost);      // D2H
cudaMemcpy(d_dest, d_src, size, cudaMemcpyDeviceToDevice);    // D2D

// Async transfers (requires pinned memory)
cudaMemcpyAsync(d_dest, h_src, size, cudaMemcpyHostToDevice, stream);

// Pinned (page-locked) host memory
float *h_ptr;
cudaMallocHost(&h_ptr, size_in_bytes);  // Faster H2D transfers
cudaFreeHost(h_ptr);

// Unified memory
float *um_ptr;
cudaMallocManaged(&um_ptr, size_in_bytes);  // Accessible by CPU and GPU
cudaFree(um_ptr);

// Prefetching (Unified Memory)
cudaMemPrefetchAsync(um_ptr, size, deviceId, stream);

// Memory advice (Unified Memory)
cudaMemAdvise(um_ptr, size, cudaMemAdviseSetReadMostly, deviceId);


/* ============================================================================
   5. MEMORY TYPES
   ============================================================================ */

// Global memory (device DRAM)
float *globalPtr;
cudaMalloc(&globalPtr, size);

// Constant memory (64KB max, cached, broadcast-optimized)
__constant__ float constData[256];
cudaMemcpyToSymbol(constData, h_data, size);
cudaMemcpyFromSymbol(h_data, constData, size);

// Shared memory (on-chip, per-block)
__shared__ float sharedData[256];           // Static
extern __shared__ float dynamicShared[];    // Dynamic (size at launch)

// Registers (automatic, per-thread)
register float localVar;  // Compiler decides

// Local memory (register spill to DRAM - avoid!)
float largeArray[1024];  // May spill to local memory


/* ============================================================================
   6. SYNCHRONIZATION
   ============================================================================ */

// Block-wide barrier
__syncthreads();              // All threads in block wait here

// Memory fences
__threadfence();              // Global memory visibility (all threads)
__threadfence_block();        // Block memory visibility
__threadfence_system();       // System-wide (including peers)

// Atomic operations (global memory)
atomicAdd(&addr, val);        // Atomic addition
atomicSub(&addr, val);        // Atomic subtraction
atomicExch(&addr, val);       // Atomic exchange
atomicMin(&addr, val);        // Atomic minimum (int)
atomicMax(&addr, val);        // Atomic maximum (int)
atomicInc(&addr, val);        // Atomic increment
atomicDec(&addr, val);        // Atomic decrement
atomicCAS(&addr, compare, val); // Compare and swap

// Atomic operations (shared memory, faster)
atomicAdd_block(&addr, val);

// Warp-level primitives (Compute Capability >= 3.0)
__shfl_sync(mask, var, srcLane);      // Shuffle from lane
__shfl_down_sync(mask, var, delta);   // Shuffle from higher lane
__shfl_up_sync(mask, var, delta);     // Shuffle from lower lane
__shfl_xor_sync(mask, var, laneMask); // Shuffle with XOR


/* ============================================================================
   7. ERROR HANDLING
   ============================================================================ */

// Check CUDA errors
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Usage
CUDA_CHECK(cudaMalloc(&ptr, size));
CUDA_CHECK(cudaGetLastError());    // Check kernel launch errors
CUDA_CHECK(cudaDeviceSynchronize()); // Check kernel execution errors


/* ============================================================================
   8. EVENTS AND TIMING
   ============================================================================ */

cudaEvent_t start, stop;
cudaEventCreate(&start);
cudaEventCreate(&stop);

cudaEventRecord(start);
// ... kernel launches ...
cudaEventRecord(stop);
cudaEventSynchronize(stop);

float elapsedMs;
cudaEventElapsedTime(&elapsedMs, start, stop);

cudaEventDestroy(start);
cudaEventDestroy(stop);


/* ============================================================================
   9. STREAMS
   ============================================================================ */

// Create stream
cudaStream_t stream;
cudaStreamCreate(&stream);
cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

// Launch kernel in stream
myKernel<<<gridSize, blockSize, sharedMem, stream>>>(...);

// Async memory transfer in stream
cudaMemcpyAsync(d_dest, h_src, size, cudaMemcpyHostToDevice, stream);

// Synchronize stream
cudaStreamSynchronize(stream);

// Wait for stream to complete (non-blocking check)
cudaError_t status = cudaStreamQuery(stream);
if (status == cudaSuccess) { /* stream is done */ }

// Stream priority
int leastPriority, greatestPriority;
cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority);
cudaStreamCreateWithPriority(&stream, cudaStreamDefault, priority);

// Destroy stream
cudaStreamDestroy(stream);


/* ============================================================================
   10. DEVICE PROPERTIES
   ============================================================================ */

cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, 0);

printf("Device: %s\n", prop.name);
printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
printf("Multiprocessors: %d\n", prop.multiProcessorCount);
printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
printf("Max threads per block: %d\n", prop.maxThreadsPerBlock);
printf("Max registers per SM: %d\n", prop.regsPerMultiprocessor);
printf("Shared memory per SM: %d KB\n", prop.sharedMemPerMultiprocessor / 1024);
printf("Warp size: %d\n", prop.warpSize);
printf("Concurrent kernels: %s\n", prop.concurrentKernels ? "Yes" : "No");


/* ============================================================================
   11. OCCUPANCY
   ============================================================================ */

// Get optimal block size for occupancy
int minGridSize, blockSize;
cudaOccupancyMaxPotentialBlockSize(&minGridSize, &blockSize, myKernel, 0, 0);

// Calculate max active blocks
int maxActiveBlocks;
cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxActiveBlocks, myKernel, blockSize, 0);


/* ============================================================================
   12. COMPILER FLAGS (nvcc)
   ============================================================================ */

// Basic compilation
nvcc -o program program.cu

// With optimization
nvcc -O3 -o program program.cu

// Specify compute capability
nvcc -arch=sm_75 -o program program.cu

// Limit registers (may increase occupancy)
nvcc -maxrregcount=32 -o program program.cu

// Generate PTX/SASS
nvcc -ptx program.cu -o program.ptx
nvcc -sass program.cu -o program.sass

// Show register usage
nvcc -ptxas-options=-v program.cu


/* ============================================================================
   13. COMMON PATTERNS
   ============================================================================ */

// Grid-stride loop (handles any array size)
__global__ void gridStrideKernel(float *data, int n) {
    int stride = blockDim.x * gridDim.x;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        data[i] = data[i] * 2.0f;
    }
}

// Parallel reduction (sum)
__global__ void reduce(float *data, float *sums, int n) {
    __shared__ float sdata[1024];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = (idx < n) ? data[idx] : 0;
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) sums[blockIdx.x] = sdata[0];
}

// Tiled matrix multiplication
#define TILE_SIZE 16
__global__ void matMul(float *A, float *B, float *C, int width) {
    __shared__ float As[TILE_SIZE][TILE_SIZE];
    __shared__ float Bs[TILE_SIZE][TILE_SIZE];
    
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    
    float sum = 0;
    for (int t = 0; t < (width + TILE_SIZE - 1) / TILE_SIZE; t++) {
        int tileCol = t * TILE_SIZE + threadIdx.x;
        int tileRow = t * TILE_SIZE + threadIdx.y;
        
        As[threadIdx.y][threadIdx.x] = (col < width && tileRow < width) 
            ? A[row * width + tileCol] : 0;
        Bs[threadIdx.y][threadIdx.x] = (tileRow < width && col < width) 
            ? B[tileRow * width + col] : 0;
        
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads();
    }
    
    if (col < width && row < width) C[row * width + col] = sum;
}


/* ============================================================================
   14. FUNCTION QUALIFIERS
   ============================================================================ */

__global__    void kernel(...)   // GPU kernel, called from host
__device__    float func(...)    // GPU function, called from GPU
__host__      void func(...)     // CPU function (normal)
__host__ __device__ void func(...)  // Compiled for both CPU and GPU
__noinline__  void func(...)     // Prevent inlining
__forceinline__ void func(...)   // Force inlining


/* ============================================================================
   15. VARIABLE QUALIFIERS
   ============================================================================ */

__device__    float globalVar;   // Global GPU variable
__constant__  float constVar;    // Constant memory (64KB limit)
__shared__    float sharedVar;   // Shared memory (per-block)
__managed__   float managedVar;  // Unified memory


/* ============================================================================
   END OF CUDA SYNTAX REFERENCE
   ============================================================================ */
