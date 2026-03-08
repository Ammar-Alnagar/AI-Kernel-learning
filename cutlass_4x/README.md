# CUTLASS 4.x Python: From High-Level Ops to Custom CuTe DSL Kernels

A complete, production-focused tutorial for mastering CUTLASS 4.x Python stack — targeting senior GPU kernel engineer roles at **NVIDIA**, **Cerebras**, and **Modular**.

---

## 🎯 Who This Is For

| Role | What You'll Build | Target Metric |
|------|-------------------|---------------|
| **NVIDIA DL Software Engineer** | Fused MHA (FlashAttention-2 parity) | Within 20% of C++ reference |
| **Cerebras LLM Inference Perf & Evals** | MoE Grouped GEMM | 2× tokens/sec vs naive loop |
| **NVIDIA Model Optimization Engineer** | FP8 Inference Pipeline | >1.5× over FP16 on Hopper |
| **Modular Kernel Engineer** | Custom Epilogue Fusion | Zero-copy torch↔cutlass bridge |

---

## 📦 Prerequisites

```bash
# Python 3.10+
pip install nvidia-cutlass-dsl

# CUDA Toolkit 12.x or 13.1 (optimal for Blackwell)
nvcc --version  # Should be >= 12.0

# GPU Targets:
# - Ampere (SM80): A100, RTX 3090/4090
# - Hopper (SM90): H100, H200
# - Blackwell (SM100/SM103): B100, B200, Thor (SM103)
```

### Verify Installation

```bash
python -c "import cutlass; print(cutlass.__version__)"
# Expected: 4.x.x

python -c "import cutlass.cute; print('CuTe DSL available')"
# Expected: CuTe DSL available
```

### Environment Variables (Recommended)

```bash
# Speeds up repeated JIT compilation
export CUTE_DSL_CACHE_DIR=$HOME/.cutlass_cache

# Enable verbose JIT output for debugging
export CUTE_DSL_VERBOSE=1

# Force specific SM architecture (if auto-detect fails)
export CUTLASS_CUDA_ARCH=sm_90
```

---

## 📚 Two-Level Programming Model

**Rule:** Always learn Level 1 first (working reference), then Level 2 (custom implementation), then benchmark both.

### LEVEL 1 — High-Level Op API (`cutlass.op`)

Use when: you want a working kernel fast, autotuning, standard ops.

```python
import cutlass
import torch

# Allocate tensors
A = torch.randn(512, 1024, dtype=torch.float16, device='cuda')
B = torch.randn(1024, 2048, dtype=torch.float16, device='cuda')
C = torch.zeros(512, 2048, dtype=torch.float16, device='cuda')

# Configure and run GEMM
plan = cutlass.op.Gemm(
    element=cutlass.float16,
    layout=cutlass.LayoutType.RowMajor
)
plan.run(A, B, C)  # C = A @ B
```

**Production use cases:**
- Standard GEMM / Conv2d / Grouped GEMM
- FP8 / BF16 / TF32 mixed precision
- Epilogue fusion (ReLU, Bias, GELU)

---

### LEVEL 2 — CuTe DSL Custom Kernels (`@cutlass.jit`)

Use when: non-standard fusion, novel attention variant, full control over pipeline and tiling.

```python
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

@cutlass.jit
def my_gemm(A: cute.Tensor, B: cute.Tensor, C: cute.Tensor):
    # Define MMA operation
    tiled_mma = cute.make_tiled_mma(
        cute.GMMA(64, 128, 32, 'f16', 'f16', 'f32'),
        cute.Layout((2, 4, 1)),  # Thread layout
    )
    
    # Define copy operations
    smem_a = cute.make_smem_tensor(A.shape, cute.float16)
    smem_b = cute.make_smem_tensor(B.shape, cute.float16)
    
    # Pipeline execution
    with cutlass.pipeline.PipelineAsync(...) as pipe:
        pipe.producer_acquire()
        # ... load data, compute, store
        pipe.producer_commit()
    
    C[:] = accum  # Write result

# Torch interop (zero-copy)
A_cutlass = from_dlpack(torch_tensor)
```

**Production use cases:**
- Custom attention variants (decode path, KV cache, int8 KV)
- Persistent kernels (SM100 tcgen05)
- Programmatic Dependent Launch (PDL)
- Blockscaled FP4 / MXFP4 (Blackwell SM103)

---

## 📁 Directory Structure

```
cutlass_4x/
├── README.md
├── setup.py                        # Install check + GPU detection + peak TFLOPS
│
├── 01_high_level_ops/              # Module 01: Core operators
│   ├── ex01_gemm_basic_FILL_IN.py
│   ├── ex02_gemm_dtypes_FILL_IN.py
│   ├── ex03_conv2d_FILL_IN.py
│   ├── ex04_grouped_gemm_FILL_IN.py
│   └── solutions/
│
├── 02_autotuning/                  # Module 02: Performance tuning
│   ├── ex01_tile_size_sweep_FILL_IN.py
│   ├── ex02_autotune_gemm_FILL_IN.py
│   └── solutions/
│
├── 03_epilogue_fusion/             # Module 03: Fused activations
│   ├── ex01_relu_epilogue_FILL_IN.py
│   ├── ex02_bias_add_epilogue_FILL_IN.py
│   ├── ex03_efc_custom_epilogue_FILL_IN.py
│   └── solutions/
│
├── 04_pytorch_integration/         # Module 04: Torch interop
│   ├── ex01_dlpack_bridge_FILL_IN.py
│   ├── ex02_custom_op_FILL_IN.py
│   ├── ex03_torch_compile_FILL_IN.py
│   └── solutions/
│
├── 05_mixed_precision/             # Module 05: Quantization
│   ├── ex01_fp8_gemm_e4m3_FILL_IN.py
│   ├── ex02_fp8_gemm_e5m2_FILL_IN.py
│   ├── ex03_int8_gemm_FILL_IN.py
│   ├── ex04_blockscaled_fp4_FILL_IN.py  # Blackwell SM100+
│   └── solutions/
│
├── 06_attention_kernels/           # Module 06: Fused MHA
│   ├── ex01_fmha_basic_FILL_IN.py
│   ├── ex02_fmha_decode_FILL_IN.py
│   ├── ex03_fmha_mixed_input_FILL_IN.py
│   └── solutions/
│
├── 07_blackwell_features/          # Module 07: SM100/SM103 features
│   ├── ex01_persistent_gemm_FILL_IN.py
│   ├── ex02_pdl_launch_FILL_IN.py
│   ├── ex03_blockscaled_gemm_FILL_IN.py
│   └── solutions/
│
└── projects/                       # Capstone projects
    ├── 01_fused_mha_full/
    │   ├── README.md
    │   ├── fmha_ampere_FILL_IN.py
    │   ├── fmha_hopper_FILL_IN.py
    │   ├── fmha_blackwell_FILL_IN.py
    │   └── benchmark.py
    │
    ├── 02_moe_grouped_gemm/
    │   ├── README.md
    │   ├── moe_gemm_FILL_IN.py
    │   └── benchmark.py
    │
    ├── 03_fp8_inference_pipeline/
    │   ├── README.md
    │   ├── fp8_linear_FILL_IN.py
    │   └── benchmark.py
    │
    └── 04_benchmarks_master/
        ├── roofline.py
        └── results/
```

---

## 🏃 How to Use This Repository

### Step 1: Run Setup Check

```bash
cd cutlass_4x
python setup.py
```

Expected output:
```
✓ nvidia-cutlass-dsl installed (version 4.x.x)
✓ CUDA Toolkit: 12.x
✓ GPU detected: NVIDIA H100 (SM90)
✓ Peak FP16 Tensor Core TFLOPS: ~989 TFLOPS (dense)
```

### Step 2: Work Through Modules in Order

Each module follows the same pattern:

1. **Read the exercise header** — understand the production use case
2. **Make predictions** — what output/performance do you expect?
3. **Fill in the TODOs** — use hints and reference links
4. **Run and verify** — does output match predictions?
5. **Profile with ncu** — identify bottlenecks
6. **(Optional) Check solutions/** — compare with reference implementation

### Step 3: Profile Every Kernel

```bash
# Example: Profile GEMM with Tensor Core metrics
ncu --metrics sm__inst_executed_pipe_tensor.sum,l2tex__t_bytes.sum \
    python 01_high_level_ops/ex01_gemm_basic_FILL_IN.py

# Example: Verify epilogue fusion (reduced global stores)
ncu --metrics l1tex__t_bytes_pipe_lsu_mem_global_op_st.sum \
    python 03_epilogue_fusion/ex01_relu_epilogue_FILL_IN.py
```

### Step 4: Build Capstone Projects

After completing Modules 01-07, tackle the capstone projects. These are designed to be **portfolio pieces** for your GitHub README and interview discussions.

---

## 📊 Benchmarking Standards

Every benchmark script should output:

```
=== Benchmark Results ===
Matrix Shape: M=4096, K=8192, N=4096
Dtype: FP16 (accum: FP32)

Performance:
  - CUTLASS GEMM:  XXX TFLOPS (YY% of peak)
  - cuBLAS:        ZZZ TFLOPS
  - torch.mm:      WWW TFLOPS

Memory:
  - Bandwidth utilized: AA GB/s (BB% of theoretical)
  - Arithmetic intensity: CC ops/byte

Latency:
  - Mean: DD ms
  - P99:  EE ms
```

---

## 🔧 Key API Reference

### JIT Compilation

```python
@cutlass.jit                          # General JIT compilation
@cutlass.cute.kernel                  # Explicit kernel marker (optional)
def my_kernel(...):
    ...
```

### Torch Interop (Zero-Copy)

```python
from cutlass.cute.runtime import from_dlpack

# torch → cutlass (no data movement)
cutlass_tensor = from_dlpack(torch_tensor)

# cutlass → torch
torch_tensor = torch.from_dlpack(cutlass_tensor)
```

### TiledCopy (Use `tv` Variant Only)

```python
# ✅ Correct (4.x API)
copy = cute.make_tiled_copy_tv(copy_atom, thr_layout, val_layout)

# ❌ Deprecated — do not use
# copy = cute.make_tiled_copy(...)
```

### Register Memory

```python
# ✅ Correct (4.x API)
rmem = cute.make_rmem_tensor_like(smem_tensor)

# ❌ Old name — do not use
# rmem = cute.make_fragment(...)
```

### Pipeline Context Managers

```python
with cutlass.pipeline.PipelineAsync(pipes, num_stages) as pipe:
    pipe.producer_acquire()
    # ... load data from global to smem
    pipe.producer_commit()
    
    pipe.consumer_wait()
    # ... compute MMA
    pipe.consumer_release()
```

### Epilogue Fusion Config (EFC) — 4.3+

```python
def my_epilogue(accum, alpha, beta, C):
    """Fused: alpha * accum + beta * C"""
    return alpha * accum + beta * C

plan = cutlass.op.Gemm(
    element=cutlass.float16,
    epilogue_functor=my_epilogue
)
```

### Blackwell-Specific Features

| Feature | SM Requirement | CUDA Version | Description |
|---------|----------------|--------------|-------------|
| `tcgen05` MMA | SM100+ | 12.5+ | 5th-gen Tensor Core gen |
| PDL | SM100+ | 12.5+ | Programmatic Dependent Launch |
| NVFP4 | SM100+ | 12.5+ | 4-bit floating point |
| MXFP4/MXFP6/MXFP8 | SM103+ | 13.0+ | Microscaling formats (Thor) |
| Blockscaled GEMM | SM103+ | 13.0+ | Per-block scaling |

---

## 🎓 Module Roadmap

| Module | Topic | Level | Production Use Case |
|--------|-------|-------|---------------------|
| **01** | High-Level Ops | 1 | Standard GEMM/Conv2d in training pipelines |
| **02** | Autotuning | 1→2 | Finding optimal tile sizes for custom shapes |
| **03** | Epilogue Fusion | 1→2 | TensorRT-LLM quantized linear layers |
| **04** | PyTorch Integration | 1→2 | Custom torch ops, torch.compile integration |
| **05** | Mixed Precision | 1→2 | FP8 inference, INT4 weights |
| **06** | Attention Kernels | 2 | FlashAttention variants, decode optimization |
| **07** | Blackwell Features | 2 | SM100 persistent kernels, PDL, FP4 |

---

## 📖 Reference Documentation

- **Official CUTLASS 4.x:** https://github.com/NVIDIA/cutlass
- **Python Examples:** https://github.com/NVIDIA/cutlass/tree/main/examples/python
- **CuTe DSL Docs:** https://nvidia.github.io/cutlass/cute_dsl/
- **CUDA Toolkit Docs:** https://docs.nvidia.com/cuda/

### ⚠️ Deprecated APIs (Do NOT Use)

| Deprecated | 4.x Equivalent |
|------------|----------------|
| `cutlass.examples.40_cutlass_py` | `examples/python/CuTeDSL/` |
| `cutlass.backend` | `nvidia-cutlass-dsl` package |
| `pycutlass` | `cutlass.op` + `cutlass.cute` |
| `make_tiled_copy` | `make_tiled_copy_tv` |
| `cutlass_cppgen` | Built-in JIT (`@cutlass.jit`) |
| `make_fragment` | `make_rmem_tensor_like` |

---

## 🏆 Capstone Project Targets

### Project 01: Fused MHA

| GPU | Target | Metric |
|-----|--------|--------|
| Ampere (SM80) | Within 20% of FlashAttention-2 C++ | Tokens/sec |
| Hopper (SM90) | Demonstrate warp-specialized pipeline | Occupancy |
| Blackwell (SM100) | PDL + persistent scheduling | Latency |

### Project 02: MoE Grouped GEMM

| Metric | Target |
|--------|--------|
| Tokens/sec | 2× vs naive expert loop |
| Memory efficiency | <10% overhead vs dense GEMM |

### Project 03: FP8 Inference Pipeline

| Metric | Target (Hopper H100) |
|--------|----------------------|
| Throughput | >1.5× over FP16 |
| Accuracy | <1% perplexity degradation |

### Project 04: Benchmarks Master

Generate auto roofline chart for every kernel:
- X-axis: Arithmetic Intensity (ops/byte)
- Y-axis: Achieved TFLOPS
- Lines: Theoretical peak, Memory bound, CUTLASS op, CuTe DSL, torch baseline

---

## 🤝 Contributing

Found a bug? Have a better implementation? Open an issue or PR.

## 📄 License

MIT License — see LICENSE file.

---

**Ready to build production-grade GPU kernels? Let's start with Module 01.**
