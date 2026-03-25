# Module 00: Setup & Your First GPU Kernel

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand what CuTile is and why it exists
- Have CuTile installed and configured
- Write and launch your first GPU kernel
- Understand the basic kernel execution model

## 📖 What is CuTile?

**CuTile** (CUDA Tile) is NVIDIA's tile-based GPU programming model designed for Tensor Cores. Unlike traditional CUDA programming that requires manual management of shared memory and thread blocks, CuTile provides a higher-level abstraction using **tiles** - immutable multidimensional data chunks that are processed in parallel.

### Key Benefits

| Traditional CUDA | CuTile |
|-----------------|--------|
| Manual thread/block management | Automatic parallelization |
| Explicit shared memory handling | Tile-based abstraction |
| Architecture-specific optimization | Portable across GPU generations |
| Complex indexing | Simple tile operations |

### The Tile Programming Model

```
┌─────────────────────────────────────────────────────────┐
│                    GPU (Global Memory)                   │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌─────────┐    │
│  │ Array A │  │ Array B │  │ Array C │  │   ...   │    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬────┘    │
│       │            │            │            │          │
│       ▼            ▼            ▼            ▼          │
│  ┌─────────────────────────────────────────────────┐    │
│  │              Tile Processing Units               │    │
│  │  ┌──────┐  ┌──────┐  ┌──────┐  ┌──────┐        │    │
│  │  │Tile 0│  │Tile 1│  │Tile 2│  │Tile 3│  ...   │    │
│  │  └──────┘  └──────┘  └──────┘  └──────┘        │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

## 🔧 Installation

### Step 1: Verify Prerequisites

```bash
# Check Python version (need 3.10+)
python --version

# Check CUDA version (need 13.1+)
nvcc --version

# Check NVIDIA driver (need r580+)
nvidia-smi
```

### Step 2: Install CuTile

```bash
pip install cuda-tile
```

### Step 3: Install Additional Dependencies

```bash
# For this tutorial, we'll use PyTorch and NumPy
pip install torch numpy cupy-cuda13x
```

### Step 4: Verify Installation

```python
import cuda.tile as ct
print(f"CuTile version: {ct.__version__}")
print(f"Available GPUs: {ct.get_device_count()}")
```

## 🏗️ CuTile Program Structure

A CuTile program has two main parts:

### 1. Host Code (Python)
- Runs on CPU
- Allocates memory (arrays)
- Launches kernels
- Uses standard Python

### 2. Tile Code (Kernel)
- Runs on GPU
- Processes tiles in parallel
- Uses CuTile operations
- Decorated with `@ct.kernel`

```
┌──────────────────┐         ┌──────────────────┐
│    Host Code     │         │    Tile Code     │
│     (Python)     │         │     (Kernel)     │
├──────────────────┤         ├──────────────────┤
│ - Allocate arrays│  launch │ - @ct.kernel     │
│ - Set grid size  │ ───────▶│ - ct.load()      │
│ - Launch kernel  │         │ - ct.store()     │
│                  │ ◀────── │ - Tile ops       │
│                  │  result │                  │
└──────────────────┘         └──────────────────┘
```

## 🚀 Your First Kernel: Vector Addition

Let's write the classic "Hello World" of GPU programming: adding two vectors!

### The Complete Example

```python
import cuda.tile as ct
import torch

TILE_SIZE = 16  # Each tile processes 16 elements

@ct.kernel
def vector_add_kernel(a, b, result):
    """Add two vectors element-wise in parallel."""
    # Get the block ID (which tile are we processing?)
    block_id = ct.bid(0)
    
    # Load tiles from global memory
    a_tile = ct.load(a, index=(block_id,), shape=(TILE_SIZE,))
    b_tile = ct.load(b, index=(block_id,), shape=(TILE_SIZE,))
    
    # Perform element-wise addition
    result_tile = a_tile + b_tile
    
    # Store result back to global memory
    ct.store(result, index=(block_id,), tile=result_tile)

# Host code
if __name__ == "__main__":
    # Create input arrays on GPU
    a = torch.randn(128, dtype=torch.float32, device='cuda')
    b = torch.randn(128, dtype=torch.float32, device='cuda')
    result = torch.zeros_like(a)
    
    # Calculate grid size (how many tiles do we need?)
    grid_size = (a.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    grid = (grid_size, 1, 1)
    
    # Launch kernel
    ct.launch(torch.cuda.current_stream(), grid, vector_add_kernel, (a, b, result))
    
    # Verify
    expected = a + b
    print(f"Results match: {torch.allclose(result, expected)}")
```

### Key Concepts Explained

#### 1. `@ct.kernel` Decorator
Marks a function as GPU code. The function cannot be called directly - it must be launched with `ct.launch()`.

#### 2. `ct.bid(dim)`
Returns the **block ID** along dimension `dim`. This tells each block which portion of data to process.

#### 3. `ct.load(array, index, shape)`
Loads data from global memory into a tile:
- `array`: The global array to load from
- `index`: Tile-space index (which tile to load)
- `shape`: The tile shape (must be power of 2!)

#### 4. `ct.store(array, index, tile)`
Stores a tile back to global memory.

#### 5. `ct.launch(stream, grid, kernel, args)`
Launches a kernel on the GPU:
- `stream`: CUDA stream for execution
- `grid`: Tuple of (x, y, z) block counts
- `kernel`: The kernel function to launch
- `args`: Arguments to pass to the kernel

## 📝 Exercise: Your Turn!

Now it's your turn to write a kernel. Open `kernel.py` and complete the exercises.

## 🧪 Testing Your Code

After completing `kernel.py`, run:
```bash
python test.py
```

If you get stuck, check `solution.py` for the reference implementation.

## 🔑 Key Takeaways

1. **CuTile** is a tile-based GPU programming model for Tensor Cores
2. **Kernels** are decorated with `@ct.kernel` and launched with `ct.launch()`
3. **Tiles** are immutable data chunks processed in parallel
4. **Block IDs** (`ct.bid()`) determine which data each block processes
5. **Load/Store** operations move data between global memory and tiles

## 📚 What's Next?

In Module 01, we'll dive deeper into the CuTile data model:
- Global Arrays vs Tiles
- Memory layouts and strides
- Tile shapes and constraints

---

**Ready to code?** Open `kernel.py` and start filling in the blanks!
