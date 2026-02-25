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

# Install all projects
install(TARGETS 
    project_01_vector_add
    project_02_gemm
    project_03_softmax
    project_04_flash_attention
    project_05_flashinfer
    DESTINATION bin
)
