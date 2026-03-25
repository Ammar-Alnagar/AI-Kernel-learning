# Module 01: Data Model - Arrays, Tiles, and Memory Layouts

## 🎯 Learning Objectives

By the end of this module, you will:
- Understand the difference between **Global Arrays** and **Tiles**
- Know how memory layouts and strides work
- Be able to query array and tile properties
- Understand tile shape constraints (power of 2)
- Master data types and dtype promotion in CuTile

## 📖 The CuTile Data Model

CuTile has a fundamentally different data model than standard Python or even CUDA:

```
┌─────────────────────────────────────────────────────────────┐
│                      HOST (CPU)                              │
│  - Standard Python code                                      │
│  - Allocates arrays via PyTorch/CuPy                         │
│  - Launches kernels                                          │
└─────────────────────────────────────────────────────────────┘
                          │
                          │ ct.launch()
                          ▼
┌─────────────────────────────────────────────────────────────┐
│                      DEVICE (GPU)                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              GLOBAL MEMORY (Arrays)                  │    │
│  │  - Mutable, strided layout                           │    │
│  │  - Accessed via ct.load() / ct.store()               │    │
│  │  - Example: torch.Tensor, cupy.ndarray               │    │
│  └─────────────────────────────────────────────────────┘    │
│                          │                                   │
│                          │ load/store                        │
│                          ▼                                   │
│  ┌─────────────────────────────────────────────────────┐    │
│  │              TILE CODE (Tiles)                       │    │
│  │  - Immutable values (like tensors in math)           │    │
│  │  - No defined storage location                       │    │
│  │  - Shape must be power of 2                          │    │
│  │  - Processed in parallel by GPU blocks               │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 🔑 Key Distinction: Arrays vs Tiles

| Property | Global Arrays | Tiles |
|----------|--------------|-------|
| **Location** | Global memory | Register/shared memory |
| **Mutability** | Mutable | Immutable |
| **Shape** | Runtime-determined | Compile-time constants |
| **Shape Constraint** | Any size | Must be power of 2 |
| **Created by** | Host (PyTorch/CuPy) | `ct.load()`, `ct.zeros()`, etc. |
| **Used in** | Host code AND tile code | Tile code ONLY |
| **Storage** | Physical memory | Abstract value |

### Visual Example

```
Global Array (128 elements)          Tiles (processed by blocks)
┌────────────────────────────────┐   
│  [0, 1, 2, 3, 4, 5, ... 127]   │   Block 0 loads tile [0:16]
└────────────────────────────────┘   ┌────────────────┐
         │                           │ Tile shape: (16)│
         │ ct.load(index=(0,))       │ Elements: 0-15  │
         ▼                           └────────────────┘
┌────────────────┐                   
│   Tile (16,)   │   Block 1 loads tile [16:32]
│ [0,1,2,...,15] │   ┌────────────────┐
└────────────────┘   │ Tile shape: (16)│
                     │ Elements: 16-31 │
                     └────────────────┘
```

## 📐 Memory Layouts and Strides

### What are Strides?

Arrays in CuTile use a **strided memory layout**. Strides determine how to map logical indices to physical memory locations.

For a 3D array with shape `(d1, d2, d3)` and strides `(s1, s2, s3)`:

```
memory_address(i1, i2, i3) = base_addr + element_size * (s1*i1 + s2*i2 + s3*i3)
```

### Example: 2D Row-Major Layout

```python
# 2D array: 4 rows x 5 columns, float32 (4 bytes per element)
# Shape: (4, 5)
# Row-major strides: (5, 1)  # Skip 5 elements for next row, 1 for next column

# Element at (row=2, col=3):
# offset = 5*2 + 1*3 = 13
# memory_address = base_addr + 4 * 13
```

### Visual Representation

```
Row-Major Layout (strides: (5, 1))
┌────┬────┬────┬────┬────┐
│  0 │  1 │  2 │  3 │  4 │  Row 0: offsets 0-4
├────┼────┼────┼────┼────┤
│  5 │  6 │  7 │  8 │  9 │  Row 1: offsets 5-9
├────┼────┼────┼────┼────┤
│ 10 │ 11 │ 12 │ 13 │ 14 │  Row 2: offsets 10-14
├────┼────┼────┼────┼────┤
│ 15 │ 16 │ 17 │ 18 │ 19 │  Row 3: offsets 15-19
└────┴────┴────┴────┴────┘

Column-Major Layout (strides: (1, 4))
┌────┬────┬────┬────┬────┐
│  0 │  4 │  8 │ 12 │ 16 │
├────┼────┼────┼────┼────┤
│  1 │  5 │  9 │ 13 │ 17 │
├────┼────┼────┼────┼────┤
│  2 │  6 │ 10 │ 14 │ 18 │
├────┼────┼────┼────┼────┤
│  3 │  7 │ 11 │ 15 │ 19 │
└────┴────┴────┴────┴────┘
```

## 🔢 Tile Shape Constraints

**CRITICAL:** Tile dimensions must be **powers of 2**!

```python
# Valid tile shapes
(16,)           # ✓ 1D, 16 = 2^4
(32, 64)        # ✓ 2D, both powers of 2
(8, 8, 16)      # ✓ 3D, all powers of 2
(128, 256)      # ✓ Large tiles for Tensor Cores

# Invalid tile shapes
(15,)           # ✗ 15 is not a power of 2
(32, 63)        # ✗ 63 is not a power of 2
(100,)          # ✗ 100 is not a power of 2
```

### Why Power of 2?

1. **Hardware Efficiency**: GPU memory accesses are optimized for power-of-2 alignments
2. **Tensor Core Requirements**: Matrix operations on Tensor Cores need specific tile sizes
3. **Address Calculation**: Power-of-2 sizes enable efficient bit-shift operations

## 📊 Data Types in CuTile

CuTile supports various data types through `cuda.tile.DType`:

### Supported Types

| Category | CuTile Dtype | Python Equivalent | Bytes |
|----------|-------------|-------------------|-------|
| **Boolean** | `ct.bool_` | `bool` | 1 |
| **Unsigned Int** | `ct.uint8`, `ct.uint16`, `ct.uint32`, `ct.uint64` | - | 1, 2, 4, 8 |
| **Signed Int** | `ct.int8`, `ct.int16`, `ct.int32`, `ct.int64` | `int` | 1, 2, 4, 8 |
| **Float 16** | `ct.float16` | - | 2 |
| **BFloat 16** | `ct.bfloat16` | - | 2 |
| **Float 32** | `ct.float32` | `float` | 4 |
| **Tensor Float 32** | `ct.tfloat32` | - | 4 |
| **Float 64** | `ct.float64` | - | 8 |
| **Float8** | `ct.float8_e4m3fn`, `ct.float8_e5m2` | - | 1 |

### Dtype Promotion Rules

When operating on tiles with different dtypes, CuTile promotes to a common dtype:

```python
# Category hierarchy: boolean < integral < floating-point

# Examples:
int32 + int32 → int32           # Same type
int32 + int64 → int64           # Wider type wins
int32 + float32 → float32       # Float wins over int
float32 + float64 → float64     # Wider float wins
```

## 🔍 Querying Properties

### Array Properties (Runtime)

```python
# Array shape is determined at runtime
array.shape  # Returns tuple of int32 scalars
array.dtype  # Returns runtime dtype
```

### Tile Properties (Compile-time)

```python
# Tile shape is known at compile time
tile.shape   # Returns compile-time constant tuple
tile.dtype   # Returns compile-time constant (e.g., ct.float32)
```

### Example

```python
import cuda.tile as ct
import torch

@ct.kernel
def example_kernel(array, output):
    block_id = ct.bid(0)
    
    # Load a tile
    tile = ct.load(array, index=(block_id,), shape=(32,))
    
    # Query tile properties (compile-time constants!)
    tile_shape = tile.shape    # (32,) - known at compile time
    tile_dtype = tile.dtype    # e.g., ct.float32
    
    # Query array properties (runtime values!)
    array_shape = array.shape  # Runtime value
    array_dtype = array.dtype  # Runtime value
    
    # ... rest of kernel
```

## 🎯 Practical Example: Understanding Tile Space

```python
import cuda.tile as ct
import torch

TILE_SIZE = 32

@ct.kernel
def process_array_kernel(input_array, output_array):
    """Process an array one tile at a time."""
    
    # Get block ID - this is a TILE SPACE index
    block_id = ct.bid(0)
    
    # When we load with index=(block_id,), we're accessing tile space
    # Block 0 → elements [0:32]
    # Block 1 → elements [32:64]
    # Block 2 → elements [64:96]
    # etc.
    
    input_tile = ct.load(input_array, index=(block_id,), shape=(TILE_SIZE,))
    
    # Process the tile
    output_tile = input_tile * 2.0 + 1.0
    
    # Store back to the same tile position
    ct.store(output_array, index=(block_id,), tile=output_tile)

# Host code
def process_array(arr: torch.Tensor) -> torch.Tensor:
    # Calculate how many tiles we need
    num_tiles = (arr.shape[0] + TILE_SIZE - 1) // TILE_SIZE
    
    output = torch.empty_like(arr)
    grid = (num_tiles, 1, 1)
    
    ct.launch(torch.cuda.current_stream(), grid, process_array_kernel, 
              (arr, output))
    return output
```

## 📝 Summary

| Concept | Key Point |
|---------|-----------|
| **Arrays** | Mutable, in global memory, any shape |
| **Tiles** | Immutable, no storage, power-of-2 shapes |
| **Strides** | Map logical indices to physical memory |
| **Tile Space** | Indexing by tiles, not individual elements |
| **Dtypes** | Homogeneous within a tile, promotion on mixed ops |

## 🧪 Exercises

Open `kernel.py` to practice:
1. Querying array and tile properties
2. Working with different tile shapes
3. Understanding stride calculations
4. Handling dtype conversions

## 📚 What's Next?

In Module 02, we'll dive deeper into kernel mechanics:
- The `@ct.kernel` decorator in detail
- Kernel parameters and constants
- Grid and block organization

---

**Ready to code?** Open `kernel.py` and start filling in the blanks!
