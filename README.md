# AI Kernel Learning

A comprehensive repository dedicated to mastering GPU kernel programming, high-performance computing (HPC) primitives, and the mathematical foundations of modern AI architectures. This project serves as a structured learning path from fundamental CUDA concepts to advanced optimizations using industry-standard libraries like CUTLASS and Triton.

---

## Repository Structure & Module Deep Dive

### 1. [CUDA Learning Path](./Cuda/) (Fundamentals to Mastery)
*Comprehensive curriculum covering the entire CUDA ecosystem, from thread hierarchy to hardware-specific optimizations.*
- **[Fundamentals](./Cuda/fundamentals/):** Core concepts including Grids, Blocks, Threads, and the Memory Hierarchy (Global vs. Shared vs. Registers).
- **[Memory Optimization](./Cuda/memory_optimization/):** Techniques for maximizing bandwidth via Coalescing, avoiding Shared Memory Bank Conflicts, and implementing Swizzling.
- **[Execution Optimization](./Cuda/execution_optimization/):** Balancing resources with Occupancy analysis and utilizing Warp-Level Primitives (`__shfl_sync`) for intra-warp communication.
- **[Advanced Concepts](./Cuda/advanced_concepts/):** Asynchronous copies (`cp.async`), Tensor Core programming, and Software Pipelining to hide memory latency.
- **[Cudalings](./Cuda/cudalings/):** A series of interactive, small-scale exercises to reinforce CUDA syntax and logic.

### 2. [Template Metaprogramming](./Template_Metaprogramming/) (Advanced C++)
*A prerequisite for advanced library development, this module covers the C++ template patterns required to understand and contribute to libraries like CUTLASS and CuTE.*

### 3. [CUTLASS 3.x Learning Path](./Cutlass_3x/) (C++ Template Library for GEMM)
*In-depth examples for CUTLASS 3.x, the foundational NVIDIA library for building high-performance, architecture-specific kernels.*
- **[Basic GEMM](./Cutlass_3x/00_basic_gemm/):** Understanding the fundamental General Matrix-Matrix Multiplication (GEMM) kernel structure.
- **[Architecture Specialization](./Cutlass_3x/):** Exploring examples tailored for Volta, Turing, Ampere, Hopper, and Blackwell architectures (e.g., `14_ampere_tf32_tensorop_gemm`, `48_hopper_warp_specialized_gemm`, `70_blackwell_gemm`).
- **[Advanced Features](./Cutlass_3x/):** Implementing Collective Operations, Epilogue Visitors, Stream-K, Sparse GEMM, Fused Operations (e.g., GEMM+Softmax), and Distributed GEMM across multiple GPUs.
- **[Python Integration](./Cutlass_3x/40_cutlass_py/):** Techniques for interfacing with CUTLASS kernels from Python.

### 4. [CuTE & CuTe-DSL Learning Path](./CuTE/) (Modern C++20 for LLM Kernels)
*A hands-on curriculum for building production-grade LLM kernels using CuTE, the modern C++20 successor to CUTLASS's layout and tensor concepts.*
- **[Module 01: Layouts](./CuTE/Module_01_Layouts/):** Mastering layout algebra (`make_layout`), hierarchical tiling, and patterns for GQA (Grouped Query Attention).
- **[Module 02: Tensors](./CuTE/Module_02_Tensors/):** Working with tensor views, per-head slicing, and block iteration with `local_tile`.
- **[Module 03-04: Tiled Operations](./CuTE/Module_03_TiledCopy/):** Using vectorized 128-bit loads, `cp.async` pipelines, and MMA Atoms for Tensor Core acceleration.
- **[Module 05-06: Pipelining & Swizzling](./CuTE/Module_05_Swizzle/):** Designing bank-conflict-free swizzling and multi-stage asynchronous pipelines.
- **[CuTe-DSL](./CuTe-DSL/):** A Python-based Domain-Specific Language for expressing CuTE concepts with greater ease and productivity.
- **[CuTile](./CuTile/):** A companion library and tutorial focused on advanced tiling strategies.
- **[Projects](./CuTE/Projects/):** Capstone implementations of a high-performance **Tiled GEMM** and the **FlashAttention-2 Prefill** kernel.

### 5. [CUTLASS 4.x & Python](./Cutlass_4x/) (High-Level Pythonic Kernel Generation)
*Exploring the next generation of CUTLASS, which emphasizes Python-first workflows and automated performance tuning.*
- **[High-Level Ops](./Cutlass_4x/01_high_level_ops/):** Defining and customizing kernels using intuitive, high-level Python constructs.
- **[Autotuning](./Cutlass_4x/02_autotuning/):** Leveraging built-in frameworks to automatically discover optimal kernel configurations for target hardware.

### 6. [Triton Learning Path](./Triton/) (Python-First Kernel Development)
*Structured modules for writing high-performance GPU kernels from scratch using the Triton DSL in Python.*
- **[Module 01-04: Basics & Tiling](./Triton/Module-01-Basics/):** Vector addition, memory operations, boundary checks, and the core philosophy of block-based programming.
- **[Module 05: Matrix Multiplication](./Triton/Module-05-Matrix-Multiplication/):** Step-by-step implementation of tiled GEMM and optimization strategies.
- **[Module 06-08: Advanced Techniques](./Triton/Module-06-Advanced-Memory/):** Cache hierarchy optimization, parallel reductions, and ensuring numerical stability.
- **[Optimization & Mastery](./Triton/optimization/):** Deep dives into pipelining, shared memory, warp specialization, and building custom attention kernels.

### 7. [Transformer Math](./transformer_math/) (Theoretical Foundations)
*The mathematical "why" behind modern AI kernels, with a focus on Large Language Model (LLM) inference.*
- **[Attention & KV Cache](./transformer_math/01_attention/):** Scaled dot-product derivations, causal masking, and the "Memory Wall" problem of the KV cache.
- **[FlashAttention](./transformer_math/05_flash_attention/):** Tiling insights, the IO problem, and the mathematical formulation of Online Softmax.
- **[Architectural Variants](./transformer_math/03_attention_variants/):** Analyzing MHA, MQA, GQA, and their hardware performance implications.
- **[Systems Analysis](./transformer_math/10_arithmetic_intensity/):** Using Roofline models to analyze decode vs. prefill performance and the impact of batch size.

### 8. [Data Structures & Algorithms](./DSA/) (Parallel Implementations)
*Classic and modern algorithms redesigned for massively parallel GPU execution.*
- **[CPU Foundations](./DSA/):** Core data structure and algorithm concepts (serving as a baseline).
- **[GPU Primitives](./GPU-DSA/):** High-performance parallel reductions, prefix sums (scans), and Morton-ordered indexing (Z-curves).
- **[GPU Sorting](./GPU-DSA/):** Radix Sort and Bitonic Sort kernel implementations.
- **[Advanced Kernels](./GPU-DSA/):** Fused LayerNorm, Online Softmax, Paged Attention (vLLM style), and Double Buffering patterns.

### 9. [Candle Learning Path](./Candle/) (Rust ML + Inference Optimization)
*A comprehensive, fill-in-the-code curriculum for Hugging Face Candle, progressing from tensor fundamentals to CUDA/Tensor Core-aware optimization and mini-LLM inference design.*
- **[Candle Workbook](./Candle/README.md):** Guided exercises covering tensors, autograd, model building, transformer blocks, KV cache, quantization, and profiling-driven optimization.
- **[Advanced Performance Track](./Candle/README.md):** Tensor Core-aware GEMM benchmarking, dtype strategy (`F16`/`BF16`), memory-layout considerations, and Nsight-based bottleneck analysis.

### 10. [Supporting Languages & Ecosystems](./)
- **[Host-Side Programming](./Concurrency/):** C++ multithreading, synchronization primitives, and driving asynchronous CUDA workloads.
- **[Python Skills](./python_crashcourse/):** A crash course and conceptual deep-dives into Python for HPC.
- **[Mojo Language](./MojoLang/):** Exploratory projects using Mojo, a new language designed for high-performance AI programming.
- **[Build, Debug, & Profile Ecosystem](./)**
    - **[CMake Build System](./Cmake-guide/):** A comprehensive guide to building complex CUDA/C++ projects with modern CMake.
    - **[PTX Assembly](./PTX/):** Low-level virtual ISA programming for fine-grained debugging and optimization.
    - **[Profiling Tools](./Profiling/):** Modules on using Nsight Compute (NCU) and Nsight Systems (NSYS) to identify and resolve performance bottlenecks.
- **[Multi-GPU Systems](./NCCL/):**
    - **[NCCL](./NCCL/):** Modules covering multi-GPU and multi-node communication primitives (AllReduce, AllGather) and the underlying ring/tree algorithms.

---

## Tooling & Environment
*   **Compilers:** `nvcc` (CUDA Toolkit 12.x+), `clang++` (C++17/20).
*   **Libraries:** CUTLASS (3.x, 4.x), CuTE, Triton, NCCL.
*   **Profiling:** NVIDIA Nsight Compute & Nsight Systems.
*   **Hardware Target:** Examples and optimizations targeting NVIDIA Ampere (SM80), Ada Lovelace (SM89), Hopper (SM90), and Blackwell (SM100-series).

## Getting Started
1.  For C++-focused development, start with **[CUDA](./Cuda/)**, then **[Template Metaprogramming](./Template_Metaprogramming/)**, followed by **[CUTLASS 3.x](./Cutlass_3x/)** and **[CuTE](./CuTE/)**.
2.  For a Python-first approach, start with **[Python Skills](./python_crashcourse/)**, then dive into the **[Triton Learning Path](./Triton/)**.
3.  Consult the **[Transformer Math](./transformer_math/)** modules to understand the theory behind the kernels you are building.
4.  Use the **[Profiling](./Profiling/)** tools continuously to measure your progress and validate optimization impact.
5.  For Rust-native model development and inference, follow the **[Candle Learning Path](./Candle/README.md)**.
