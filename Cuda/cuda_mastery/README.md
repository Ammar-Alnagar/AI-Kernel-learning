# CUDA Mastery Path

A structured journey from CUDA basics to advanced optimization techniques.

## 📚 Curriculum

### 1. **Basics** (`01_basics/`)
- Hello World CUDA
- Vector Addition
- Thread indexing (threadIdx, blockIdx, blockDim, gridDim)
- 1D, 2D, 3D grid/block configurations

### 2. **Memory Model** (`02_memory_model/`)
- Global memory
- Constant memory
- Local memory
- Memory coalescing patterns
- Memory bandwidth optimization

### 3. **Shared Memory** (`03_shared_memory/`)
- Shared memory basics
- Tiled matrix multiplication
- Bank conflicts and avoidance
- Dynamic shared memory

### 4. **Synchronization** (`04_synchronization/`)
- `__syncthreads()`
- Cooperative groups
- Atomic operations
- Warp-level primitives

### 5. **Optimization** (`05_optimization/`)
- Occupancy tuning
- Register usage
- Instruction-level parallelism
- Profiling with nvprof/nsys

### 6. **Advanced** (`06_advanced/`)
- CUDA Streams
- Multi-GPU programming
- Unified Memory
- Graph API

## 🚀 How to Use

1. Start from `01_basics/` and work sequentially
2. Each lesson contains:
   - `.cu` file with code examples
   - Exercises to complete
   - Key concepts explained in comments
3. Compile with: `nvcc -o <name> <file>.cu`
4. Run with: `./<name>`

## 🔧 Prerequisites

- Basic C/C++ knowledge
- NVIDIA GPU with CUDA support
- CUDA Toolkit installed

## 📖 Quick Start

```bash
cd 01_basics
nvcc -o hello hello.cu
./hello
```
