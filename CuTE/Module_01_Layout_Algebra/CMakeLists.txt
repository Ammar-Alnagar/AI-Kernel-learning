cmake_minimum_required(VERSION 3.20)
project(LayoutStudy LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

find_package(CUDAToolkit REQUIRED)

set(CUTLASS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../../third_party/cutlass/include"
    CACHE PATH "Path to CUTLASS include directory")

if(NOT EXISTS "${CUTLASS_INCLUDE_DIR}")
    message(FATAL_ERROR "CUTLASS not found at: ${CUTLASS_INCLUDE_DIR}")
endif()

add_executable(layout_study layout_study.cu)

target_include_directories(layout_study PRIVATE
    ${CUTLASS_INCLUDE_DIR}
    ${CUDAToolkit_INCLUDE_DIRS}
)

target_link_libraries(layout_study PRIVATE CUDA::cudart)

install(TARGETS layout_study DESTINATION bin)
