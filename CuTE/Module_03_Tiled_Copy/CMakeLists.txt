cmake_minimum_required(VERSION 3.20)
project(TiledCopyBasics LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (sm_89 for RTX 4060)
set(CMAKE_CUDA_ARCHITECTURES "89")

# Set extended constexpr support for CuTe
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Include path for CuTe (CUTLASS submodule)
set(CUTLASS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../cutlass/include" CACHE PATH "Path to CUTLASS include directory")

if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR "CUTLASS include directory not found. Please set CUTLASS_INCLUDE_DIR.")
else()
    include_directories(${CUTLASS_INCLUDE_DIR})
endif()

# Module 03: Tiled Copy
add_executable(tiled_copy_basics tiled_copy_basics.cu)

# Installation rules
install(TARGETS tiled_copy_basics DESTINATION bin)