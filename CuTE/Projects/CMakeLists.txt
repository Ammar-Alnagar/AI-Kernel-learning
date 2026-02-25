# Projects Module - Hands-on CuTe/CUTLASS Kernel Implementation
# Each project can be built individually using: cmake --build . --target <project_name>

cmake_minimum_required(VERSION 3.20)
project(CuTe_Projects LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../cutlass/include"
    CACHE PATH "Path to CUTLASS include directory")

if(NOT EXISTS "${CUTLASS_INCLUDE_DIR}")
    message(FATAL_ERROR "CUTLASS not found at: ${CUTLASS_INCLUDE_DIR}")
endif()

# Project 01: Vector Add - Introduction to CuTe tensors and element-wise operations
add_executable(project_01_vector_add 01_vector_add/vector_add.cu)
target_include_directories(project_01_vector_add PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_01_vector_add PRIVATE CUDA::cudart)

# Project 02: GEMM - General Matrix Multiply using CuTe MMA atoms
add_executable(project_02_gemm 02_gemm/gemm.cu)
target_include_directories(project_02_gemm PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_02_gemm PRIVATE CUDA::cudart)

# Project 03: Softmax - Row-wise softmax using CuTe reduction patterns
add_executable(project_03_softmax 03_softmax/softmax.cu)
target_include_directories(project_03_softmax PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_03_softmax PRIVATE CUDA::cudart)

# Project 04: FlashAttention - Attention mechanism with tiling
add_executable(project_04_flash_attention 04_flash_attention/flash_attention.cu)
target_include_directories(project_04_flash_attention PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_04_flash_attention PRIVATE CUDA::cudart)

# Project 05: FlashInfer - Variable sequence length attention
add_executable(project_05_flashinfer 05_flashinfer/flashinfer.cu)
target_include_directories(project_05_flashinfer PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_05_flashinfer PRIVATE CUDA::cudart)

# Project 06: Tiled GEMM with Shared Memory Optimization
add_executable(project_06_tiled_gemm_smem 06_tiled_gemm_smem/tiled_gemm_smem.cu)
target_include_directories(project_06_tiled_gemm_smem PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_06_tiled_gemm_smem PRIVATE CUDA::cudart)

# Project 07: MMA GEMM with Tensor Cores
add_executable(project_07_mma_gemm 07_mma_gemm/mma_gemm.cu)
target_include_directories(project_07_mma_gemm PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_07_mma_gemm PRIVATE CUDA::cudart)

# Project 08: Pipelined GEMM with Async Copy
add_executable(project_08_pipelined_gemm 08_pipelined_gemm/pipelined_gemm.cu)
target_include_directories(project_08_pipelined_gemm PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_08_pipelined_gemm PRIVATE CUDA::cudart)

# Project 09: Vectorized Copy Kernel
add_executable(project_09_vectorized_copy 09_vectorized_copy/vectorized_copy.cu)
target_include_directories(project_09_vectorized_copy PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_09_vectorized_copy PRIVATE CUDA::cudart)

# Project 10: Multi-Head Attention with KV-Cache
add_executable(project_10_mha_kv_cache 10_mha_kv_cache/mha_kv_cache.cu)
target_include_directories(project_10_mha_kv_cache PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_10_mha_kv_cache PRIVATE CUDA::cudart)

# Project 11: INT8 GEMM with Fused Dequantization
add_executable(project_11_int8_gemm 11_int8_gemm/int8_gemm.cu)
target_include_directories(project_11_int8_gemm PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_11_int8_gemm PRIVATE CUDA::cudart)

# Project 12: FP8 GEMM
add_executable(project_12_fp8_gemm 12_fp8_gemm/fp8_gemm.cu)
target_include_directories(project_12_fp8_gemm PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_12_fp8_gemm PRIVATE CUDA::cudart)

# Project 13: Fused GEMM + RoPE (Rotary Position Embedding)
add_executable(project_13_gemm_rope 13_gemm_rope/gemm_rope.cu)
target_include_directories(project_13_gemm_rope PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_13_gemm_rope PRIVATE CUDA::cudart)

# Project 14: Fused MLA (Multi-head Latent Attention)
add_executable(project_14_mla 14_mla/mla.cu)
target_include_directories(project_14_mla PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(project_14_mla PRIVATE CUDA::cudart)

# Install all projects
install(TARGETS
    project_01_vector_add
    project_02_gemm
    project_03_softmax
    project_04_flash_attention
    project_05_flashinfer
    project_06_tiled_gemm_smem
    project_07_mma_gemm
    project_08_pipelined_gemm
    project_09_vectorized_copy
    project_10_mha_kv_cache
    project_11_int8_gemm
    project_12_fp8_gemm
    project_13_gemm_rope
    project_14_mla
    DESTINATION bin
)
