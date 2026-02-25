# CuTe Tutorials CMakeLists.txt
# Build all tutorial examples for learning CuTe

cmake_minimum_required(VERSION 3.20)
project(CuTe_Tutorials LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (sm_80 for broad compatibility, adjust as needed)
set(CMAKE_CUDA_ARCHITECTURES "80;86;89" CACHE STRING "CUDA architectures to build for")

# Enable extended constexpr support for CuTe
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Include path for CuTe (CUTLASS)
set(CUTLASS_INCLUDE_DIR "${PROJECT_SOURCE_DIR}/../cutlass/include" CACHE PATH "Path to CUTLASS include directory")

if(NOT EXISTS ${CUTLASS_INCLUDE_DIR})
    message(WARNING "CUTLASS include directory not found at ${CUTLASS_INCLUDE_DIR}")
    message(WARNING "Please ensure CUTLASS submodule is initialized: git submodule update --init --recursive")
else()
    message(STATUS "Using CUTLASS from: ${CUTLASS_INCLUDE_DIR}")
endif()

include_directories(${CUTLASS_INCLUDE_DIR})

# Enable CUDA separable compilation for device code
set_property(GLOBAL PROPERTY CUDA_SEPARABLE_COMPILATION ON)

# ============================================================================
# Module 01: Introduction to Layout Algebra
# ============================================================================
add_executable(tutorial_01_layout_basics 
    01_Intro_Layout_Algebra/examples/01_layout_basics.cu
)
target_include_directories(tutorial_01_layout_basics PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_01_shape_stride 
    01_Intro_Layout_Algebra/examples/02_shape_and_stride.cu
)
target_include_directories(tutorial_01_shape_stride PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_01_composition 
    01_Intro_Layout_Algebra/examples/03_layout_composition.cu
)
target_include_directories(tutorial_01_composition PRIVATE ${CUTLASS_INCLUDE_DIR})

# ============================================================================
# Module 02: CuTe Tensors
# ============================================================================
add_executable(tutorial_02_tensor_creation 
    02_CuTe_Tensors/examples/01_tensor_creation.cu
)
target_include_directories(tutorial_02_tensor_creation PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_02_tensor_views 
    02_CuTe_Tensors/examples/02_tensor_views.cu
)
target_include_directories(tutorial_02_tensor_views PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_02_tensor_arithmetic 
    02_CuTe_Tensors/examples/03_tensor_arithmetic.cu
)
target_include_directories(tutorial_02_tensor_arithmetic PRIVATE ${CUTLASS_INCLUDE_DIR})

# ============================================================================
# Module 03: Tiled Copy
# ============================================================================
add_executable(tutorial_03_copy_basics 
    03_Tiled_Copy/examples/01_copy_basics.cu
)
target_include_directories(tutorial_03_copy_basics PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_03_tiled_copy 
    03_Tiled_Copy/examples/02_tiled_copy_2d.cu
)
target_include_directories(tutorial_03_tiled_copy PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_03_thread_mapping 
    03_Tiled_Copy/examples/03_thread_mapping.cu
)
target_include_directories(tutorial_03_thread_mapping PRIVATE ${CUTLASS_INCLUDE_DIR})

# ============================================================================
# Module 04: MMA Atoms
# ============================================================================
add_executable(tutorial_04_mma_introduction 
    04_MMA_Atoms/examples/01_mma_introduction.cu
)
target_include_directories(tutorial_04_mma_introduction PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_04_mma_shapes 
    04_MMA_Atoms/examples/02_mma_shapes.cu
)
target_include_directories(tutorial_04_mma_shapes PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_04_mma_accumulation 
    04_MMA_Atoms/examples/03_mma_accumulation.cu
)
target_include_directories(tutorial_04_mma_accumulation PRIVATE ${CUTLASS_INCLUDE_DIR})

# ============================================================================
# Module 05: Shared Memory
# ============================================================================
add_executable(tutorial_05_shared_basics 
    05_Shared_Memory/examples/01_shared_memory_basics.cu
)
target_include_directories(tutorial_05_shared_basics PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_05_swizzling 
    05_Shared_Memory/examples/02_swizzling.cu
)
target_include_directories(tutorial_05_swizzling PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_05_bank_conflict 
    05_Shared_Memory/examples/03_bank_conflict_free.cu
)
target_include_directories(tutorial_05_bank_conflict PRIVATE ${CUTLASS_INCLUDE_DIR})

# ============================================================================
# Module 06: Advanced Patterns
# ============================================================================
add_executable(tutorial_06_gemm_pattern 
    06_Advanced_Patterns/examples/01_gemm_pattern.cu
)
target_include_directories(tutorial_06_gemm_pattern PRIVATE ${CUTLASS_INCLUDE_DIR})

add_executable(tutorial_06_pipeline 
    06_Advanced_Patterns/examples/02_software_pipeline.cu
)
target_include_directories(tutorial_06_pipeline PRIVATE ${CUTLASS_INCLUDE_DIR})

# ============================================================================
# Exercise Solutions (Optional - can be excluded from build)
# ============================================================================
option(BUILD_SOLUTIONS "Build exercise solutions" ON)

if(BUILD_SOLUTIONS)
    # Module 01 Solutions
    add_executable(exercise_01_solution 
        01_Intro_Layout_Algebra/solutions/exercise_01_solution.cu
    )
    target_include_directories(exercise_01_solution PRIVATE ${CUTLASS_INCLUDE_DIR})

    # Module 02 Solutions
    add_executable(exercise_02_solution 
        02_CuTe_Tensors/solutions/exercise_02_solution.cu
    )
    target_include_directories(exercise_02_solution PRIVATE ${CUTLASS_INCLUDE_DIR})

    # Module 03 Solutions
    add_executable(exercise_03_solution 
        03_Tiled_Copy/solutions/exercise_03_solution.cu
    )
    target_include_directories(exercise_03_solution PRIVATE ${CUTLASS_INCLUDE_DIR})

    # Module 04 Solutions
    add_executable(exercise_04_solution 
        04_MMA_Atoms/solutions/exercise_04_solution.cu
    )
    target_include_directories(exercise_04_solution PRIVATE ${CUTLASS_INCLUDE_DIR})

    # Module 05 Solutions
    add_executable(exercise_05_solution 
        05_Shared_Memory/solutions/exercise_05_solution.cu
    )
    target_include_directories(exercise_05_solution PRIVATE ${CUTLASS_INCLUDE_DIR})
endif()

# ============================================================================
# Print available targets
# ============================================================================
message(STATUS "")
message(STATUS "=== CuTe Tutorials Build Configuration ===")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Solutions: ${BUILD_SOLUTIONS}")
message(STATUS "")
message(STATUS "Available tutorial targets:")
message(STATUS "  Module 01: tutorial_01_layout_basics, tutorial_01_shape_stride, tutorial_01_composition")
message(STATUS "  Module 02: tutorial_02_tensor_creation, tutorial_02_tensor_views, tutorial_02_tensor_arithmetic")
message(STATUS "  Module 03: tutorial_03_copy_basics, tutorial_03_tiled_copy, tutorial_03_thread_mapping")
message(STATUS "  Module 04: tutorial_04_mma_introduction, tutorial_04_mma_shapes, tutorial_04_mma_accumulation")
message(STATUS "  Module 05: tutorial_05_shared_basics, tutorial_05_swizzling, tutorial_05_bank_conflict")
message(STATUS "  Module 06: tutorial_06_gemm_pattern, tutorial_06_pipeline")
if(BUILD_SOLUTIONS)
    message(STATUS "  Solutions: exercise_01_solution through exercise_05_solution")
endif()
message(STATUS "")
