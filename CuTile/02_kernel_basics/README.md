# Module 02: Kernel Basics - @kernel Decorator, Launch, and Block IDs

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the `@ct.kernel` decorator and its options
- Know how to properly launch kernels with `ct.launch()`
- Master block ID retrieval with `ct.bid()` and grid organization
- Understand kernel parameters and compile-time constants
- Learn about `ct.cdiv()` and `ct.num_blocks()`

## 📖 Kernel Structure Overview

A CuTile kernel is a special function that runs on the GPU:

```python
import cuda.tile as ct

@ct.kernel
def my_kernel(arg1, arg2, output):
    # This code runs on the GPU
    block_id = ct.bid(0)
    # ... process data ...
```

### Key Rules

| Rule | Description |
|------|-------------|
| **Cannot call directly** | Kernels must be launched with `ct.launch()` |
| **Tile code only** | Kernel body uses CuTile operations, not standard Python |
| **No returns** | Kernels write results to output arrays, not return values |
| **Parameters** | Only arrays and compile-time constants can be parameters |

## 🔨 The @kernel Decorator

### Basic Usage

```python
@ct.kernel
def simple_kernel(input_array, output_array):
    block_id = ct.bid(0)
    tile = ct.load(input_array, index=(block_id,), shape=(32,))
    ct.store(output_array, index=(block_id,), tile=tile)
```

### Advanced: num_ctas Parameter

The `num_ctas` parameter controls how many CTAs (Cooperative Thread Arrays) run per block:

```python
# Let the compiler decide based on target architecture
@ct.kernel(num_ctas=ct.ByTarget(sm_100=2))
def optimized_kernel(input_array, output_array):
    # This kernel will use 2 CTAs per block on SM 100 (Blackwell)
    pass

# Explicit number
@ct.kernel(num_ctas=2)
def multi_cta_kernel(input_array, output_array):
    pass
```

## 🚀 Launching Kernels with ct.launch()

### Syntax

```python
ct.launch(stream, grid, kernel, args)
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `stream` | CUDA Stream | Execution stream (e.g., `torch.cuda.current_stream()`) |
| `grid` | tuple | 3-tuple of (x, y, z) block counts |
| `kernel` | function | The kernel function to launch |
| `args` | tuple | Arguments to pass to the kernel |

### Complete Example

```python
import cuda.tile as ct
import torch
from math import ceil

TILE_SIZE = 32

@ct.kernel
def scale_kernel(array, output, scale_factor):
    block_id = ct.bid(0)
    tile = ct.load(array, index=(block_id,), shape=(TILE_SIZE,))
    result_tile = tile * scale_factor
    ct.store(output, index=(block_id,), tile=result_tile)

def scale_array(array: torch.Tensor, scale_factor: float) -> torch.Tensor:
    # Create output
    output = torch.empty_like(array)
    
    # Calculate grid size
    num_blocks = ceil(array.shape[0] / TILE_SIZE)
    grid = (num_blocks, 1, 1)
    
    # Launch kernel
    ct.launch(torch.cuda.current_stream(), grid, scale_kernel, 
              (array, output, scale_factor))
    
    return output
```

## 🎯 Block IDs with ct.bid()

### Getting Block IDs

```python
# Get block ID along dimension 0 (most common)
block_id_x = ct.bid(0)

# Get block ID along dimension 1
block_id_y = ct.bid(1)

# Get block ID along dimension 2
block_id_z = ct.bid(2)
```

### 1D Grid Example

```python
@ct.kernel
def process_1d(array, output):
    # Single dimension: block_id ranges from 0 to grid[0]-1
    block_id = ct.bid(0)
    
    # Each block processes one tile
    tile = ct.load(array, index=(block_id,), shape=(32,))
    # ... process ...
```

### 2D Grid Example

```python
@ct.kernel
def process_2d(matrix, output):
    # Two dimensions: useful for 2D data like images/matrices
    block_id_x = ct.bid(0)  # Column index
    block_id_y = ct.bid(1)  # Row index
    
    # Load a 2D tile
    tile = ct.load(matrix, index=(block_id_x, block_id_y), shape=(32, 32))
    # ... process ...
```

### 3D Grid Example

```python
@ct.kernel
def process_3d(volume, output):
    # Three dimensions: useful for 3D data or batched operations
    block_id_x = ct.bid(0)  # Batch index
    block_id_y = ct.bid(1)  # Row index  
    block_id_z = ct.bid(2)  # Column index
    
    # Load a 3D tile
    tile = ct.load(volume, index=(block_id_x, block_id_y, block_id_z), 
                   shape=(1, 32, 32))
    # ... process ...
```

## 📐 Grid Size Calculation

### Using ct.cdiv() (Ceiling Division)

CuTile provides `ct.cdiv()` for compile-time ceiling division:

```python
# Calculate blocks needed: ceil(size / tile_size)
num_blocks = ct.cdiv(array_size, tile_size)

# In host code (Python), use math.ceil or integer arithmetic
from math import ceil
num_blocks = ceil(array_size / tile_size)
# Or: num_blocks = (array_size + tile_size - 1) // tile_size
```

### Using ct.num_blocks()

Get the number of blocks along an axis at runtime:

```python
@ct.kernel
def adaptive_kernel(array, output):
    # Get number of blocks along axis 0
    num_blocks_x = ct.num_blocks(0)
    
    # Get current block ID
    block_id = ct.bid(0)
    
    # Can use for bounds checking or adaptive processing
    if block_id < num_blocks_x:
        # Process this block's data
        pass
```

### Using ct.num_tiles()

Get the number of tiles in an array's tile space:

```python
@ct.kernel
def tile_aware_kernel(array, output):
    # Get number of tiles along axis 1 with shape (32, 64)
    num_tiles = ct.num_tiles(array, axis=1, shape=(32, 64))
    
    # Loop over all tiles
    for i in range(num_tiles):
        tile = ct.load(array, index=(i,), shape=(32, 64))
        # ... process ...
```

## 🔢 Compile-Time Constants

### Using ct.Constant

For values that must be known at compile time:

```python
# Define a constant type alias
ConstInt = ct.Constant[int]

@ct.kernel
def kernel_with_constants(array, output, 
                          tile_size: ConstInt,
                          scale: ConstInt):
    # tile_size and scale are compile-time constants
    # Can be used in shape specifications
    tile = ct.load(array, index=(ct.bid(0),), shape=(tile_size,))
    result = tile * scale
    ct.store(output, index=(ct.bid(0),), tile=result)

# Launch with constant arguments
ct.launch(stream, grid, kernel_with_constants, 
          (array, output, 32, 2))
```

### Why Constants Matter

Constants enable:
- **Compile-time optimization**: The compiler can optimize based on known values
- **Shape specifications**: Tile shapes must be compile-time constants
- **Loop unrolling**: Loops with constant bounds can be unrolled

## 🎯 Complete Example: 2D Matrix Processing

```python
import cuda.tile as ct
import torch
from math import ceil

ConstInt = ct.Constant[int]

@ct.kernel
def matrix_scale_kernel(matrix, output, 
                        tile_m: ConstInt, tile_n: ConstInt,
                        scale: float):
    """
    Scale a 2D matrix using 2D grid.
    
    Each block processes one tile of the matrix.
    """
    # Get 2D block coordinates
    block_id_m = ct.bid(0)  # Row block index
    block_id_n = ct.bid(1)  # Column block index
    
    # Load a 2D tile
    tile = ct.load(matrix, index=(block_id_m, block_id_n), 
                   shape=(tile_m, tile_n))
    
    # Scale the tile
    result_tile = tile * scale
    
    # Store back
    ct.store(output, index=(block_id_m, block_id_n), tile=result_tile)

def scale_matrix(matrix: torch.Tensor, scale: float) -> torch.Tensor:
    """Host function to scale a 2D matrix."""
    
    # Tile sizes (must be powers of 2)
    TILE_M = 32
    TILE_N = 32
    
    # Create output
    output = torch.empty_like(matrix)
    
    # Calculate 2D grid
    grid_m = ceil(matrix.shape[0] / TILE_M)
    grid_n = ceil(matrix.shape[1] / TILE_N)
    grid = (grid_m, grid_n, 1)
    
    # Launch kernel
    ct.launch(torch.cuda.current_stream(), grid, matrix_scale_kernel,
              (matrix, output, TILE_M, TILE_N, scale))
    
    return output
```

## 📊 Grid Organization Patterns

### Pattern 1: 1D Grid for 1D Data

```python
# Vector operations
grid = (num_elements // TILE_SIZE + 1, 1, 1)
block_id = ct.bid(0)
```

### Pattern 2: 2D Grid for 2D Data

```python
# Matrix operations
grid = (rows // TILE_ROWS + 1, cols // TILE_COLS + 1, 1)
block_id_x = ct.bid(0)
block_id_y = ct.bid(1)
```

### Pattern 3: 3D Grid for Batched Operations

```python
# Batched matrix operations
grid = (batch_size, rows // TILE_ROWS + 1, cols // TILE_COLS + 1)
batch_id = ct.bid(0)
row_id = ct.bid(1)
col_id = ct.bid(2)
```

## 📝 Summary

| Concept | Key Point |
|---------|-----------|
| **@ct.kernel** | Marks GPU functions, cannot be called directly |
| **ct.launch()** | Launches kernels with (stream, grid, kernel, args) |
| **ct.bid(dim)** | Gets block ID along dimension |
| **ct.cdiv(a, b)** | Ceiling division for grid calculation |
| **ct.num_blocks(axis)** | Get number of blocks along axis |
| **ct.num_tiles()** | Get number of tiles in array's tile space |
| **ct.Constant** | Type for compile-time constant parameters |

## 🧪 Exercises

Open `kernel.py` to practice:
1. Writing kernels with different grid dimensions
2. Using compile-time constants
3. Calculating grid sizes with ct.cdiv
4. Multi-dimensional block ID handling

## 📚 What's Next?

In Module 03, we'll dive deep into memory operations:
- `ct.load()` and `ct.store()` in detail
- Padding modes for boundary handling
- `ct.gather()` and `ct.scatter()` for indexed access

---

**Ready to code?** Open `kernel.py` and start filling in the blanks!
