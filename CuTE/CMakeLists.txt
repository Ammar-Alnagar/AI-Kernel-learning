cmake_minimum_required(VERSION 3.20)
project(CuTe_Learning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (sm_89 for RTX 4060)
set(CMAKE_CUDA_ARCHITECTURES "89")

# Set extended constexpr support for CuTe
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Include path for CuTe (CUTLASS submodule)
set(CUTLASS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/cutlass/include" CACHE PATH "Path to CUTLASS include directory")

if(NOT CUTLASS_INCLUDE_DIR)
    message(FATAL_ERROR "CUTLASS include directory not found. Please set CUTLASS_INCLUDE_DIR.")
else()
    include_directories(${CUTLASS_INCLUDE_DIR})
endif()

# Common CUDA properties
set_property(TARGET PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# Module 01: Layout Algebra
add_executable(layout_study Module_01_Layout_Algebra/layout_study.cu)
target_include_directories(layout_study PRIVATE ${CUTLASS_INCLUDE_DIR})

# Module 02: CuTe Tensors
add_executable(tensor_basics Module_02_CuTe_Tensors/tensor_basics.cu)
target_include_directories(tensor_basics PRIVATE ${CUTLASS_INCLUDE_DIR})

# Module 03: Tiled Copy
add_executable(tiled_copy_basics Module_03_Tiled_Copy/tiled_copy_basics.cu)
target_include_directories(tiled_copy_basics PRIVATE ${CUTLASS_INCLUDE_DIR})

# Module 04: MMA Atoms
add_executable(mma_atom_basics Module_04_MMA_Atoms/mma_atom_basics.cu)
target_include_directories(mma_atom_basics PRIVATE ${CUTLASS_INCLUDE_DIR})

# Module 05: Shared Memory Swizzling
add_executable(shared_memory_layouts Module_05_Shared_Memory_Swizzling/shared_memory_layouts.cu)
target_include_directories(shared_memory_layouts PRIVATE ${CUTLASS_INCLUDE_DIR})

# Module 06: Collective Mainloops
add_executable(producer_consumer_pipeline Module_06_Collective_Mainloops/producer_consumer_pipeline.cu)
target_include_directories(producer_consumer_pipeline PRIVATE ${CUTLASS_INCLUDE_DIR})

# Projects Module - Hands-on kernel implementations
add_subdirectory(Projects)

# Installation rules
install(TARGETS 
    layout_study 
    tensor_basics 
    tiled_copy_basics 
    mma_atom_basics 
    shared_memory_layouts 
    producer_consumer_pipeline 
    DESTINATION bin
)