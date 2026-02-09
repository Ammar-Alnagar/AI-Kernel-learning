# CUTLASS Learning Modules

This repository contains a series of modules to learn CUTLASS (CUDA Templates for Linear Algebra Subroutines).

## Modules

1. **Module 1: Layouts and Tensors** - Introduction to CUTLASS layouts and tensor representations
2. **Module 2: Tiled Copy** - Understanding tiled memory operations
3. **Module 3: Tiled MMA** - Matrix multiply-accumulate operations
4. **Module 4: Fused Bias-Add** - Fusing bias addition with GEMM operations
5. **Module 5: Mainloop Pipelining** - Temporal overlap and throughput optimization
6. **Module 6: Fused Epilogues** - Functional fusion to avoid VRAM roundtrips

## Prerequisites

- CUDA Toolkit (with support for sm_89 architecture)
- CMake 3.20 or later
- A compatible GPU (RTX 40 series or similar with Ada Lovelace architecture)

## Building

### Build Individual Modules

Each module has its own build script:

```bash
# Build module 1
cd module1-Layouts\ and\ Tensors
./build.sh

# Build module 2
cd ../module2-Tiled\ Copy
./build.sh

# And so on for other modules...
```

### Build All Modules

To build all modules at once:

```bash
./build_all.sh
```

## Running Modules

After building, each module will produce an executable that can be run directly:

```bash
# From within the module's build directory
./module1    # For module 1
./module2    # For module 2
./module3_main  # For module 3
./mma_atom_spatial  # For module 4
./module5    # For module 5
./module6    # For module 6
```

## Project Structure

```
Cutlass3.x/
├── CMakeLists.txt          # Top-level CMake configuration
├── build_all.sh            # Script to build all modules
├── module1-Layouts and Tensors/
│   ├── CMakeLists.txt      # Module-specific CMake configuration
│   ├── build.sh            # Module-specific build script
│   ├── main.cu             # Module source code
│   └── README.md           # Module documentation
├── module2-Tiled Copy/
│   ├── CMakeLists.txt
│   ├── build.sh
│   ├── main.cu
│   └── README.md
├── ...
└── third_party/cutlass/    # CUTLASS library submodule
```

## Notes

- Make sure to initialize the CUTLASS submodule: `git submodule update --init --recursive`
- All modules are configured to target sm_89 architecture (Ada Lovelace GPUs)
- Each module can be built independently or as part of the whole project