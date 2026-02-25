# Projects Build Guide

## Quick Start

### Build All Projects from Root

```bash
cd /path/to/CuTE
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Build Single Project from Root

```bash
cd /path/to/CuTE/build
make project_01_vector_add
make project_07_mma_gemm
make project_13_gemm_rope
```

### Build Project Standalone (Recommended for Development)

```bash
# Navigate to specific project
cd Projects/01_vector_add

# Build in isolated directory
mkdir -p build && cd build
cmake ..
make -j$(nproc)

# Run
./project_01_vector_add
```

## Build Targets

| Target Name | Project |
|-------------|---------|
| `project_01_vector_add` | Vector Add |
| `project_02_gemm` | Basic GEMM |
| `project_03_softmax` | Softmax |
| `project_04_flash_attention` | FlashAttention |
| `project_05_flashinfer` | FlashInfer |
| `project_06_tiled_gemm_smem` | Tiled GEMM + Shared Memory |
| `project_07_mma_gemm` | MMA GEMM + Tensor Cores |
| `project_08_pipelined_gemm` | Pipelined GEMM |
| `project_09_vectorized_copy` | Vectorized Copy |
| `project_10_mha_kv_cache` | Multi-Head Attention + KV-Cache |
| `project_11_int8_gemm` | INT8 GEMM |
| `project_12_fp8_gemm` | FP8 GEMM |
| `project_13_gemm_rope` | Fused GEMM + RoPE |
| `project_14_mla` | MLA (Multi-head Latent Attention) |

## Requirements

- CUDA 12.x
- CMake 3.20+
- CUTLASS (included in repository)
- GPU with compute capability 8.9 (RTX 4060) or compatible

## Troubleshooting

### CUTLASS Not Found

Ensure the cutlass submodule is initialized:
```bash
git submodule update --init --recursive
```

### Build Errors

Try cleaning and rebuilding:
```bash
rm -rf build/
mkdir build && cd build
cmake ..
make -j$(nproc)
```

### Runtime Errors

Ensure you're running on a compatible GPU (sm_89 architecture recommended).
