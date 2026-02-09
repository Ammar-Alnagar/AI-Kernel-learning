cmake_minimum_required(VERSION 3.18)

project(CutlassLearning LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Set CUDA architectures (sm_89 for Ada Lovelace RTX 40 series)
set(CMAKE_CUDA_ARCHITECTURES "89")

# Add CUTLASS as a submodule
# Note: This assumes you've run `git submodule update --init --recursive`
set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)

# Configure CUTLASS options before including it
set(CUTLASS_NVCC_ARCHS_SUPPORTED "80;86;87;89;90")
set(CUTLASS_LIBRARY_KERNELS "")
set(BUILD_TOOLS OFF CACHE BOOL "Build CUTLASS tools")
set(CUTLASS_UNITY_BUILD_ENABLED ON CACHE BOOL "Enable Unity/Jumbo builds")

# Include CUTLASS
if(EXISTS ${CUTLASS_DIR})
    add_subdirectory(${CUTLASS_DIR} cutlass-build)
else()
    message(FATAL_ERROR "CUTLASS submodule not found. Please run: git submodule update --init --recursive")
endif()

# Common CUDA compiler flags
set(COMMON_NVCC_FLAGS 
    -arch=sm_89
    -O3
    -use_fast_math
    -lineinfo
    -DNDEBUG
)

# Function to create a module executable
function(add_module NAME SOURCE_DIR)
    set(SRC_FILE ${SOURCE_DIR}/main.cu)
    
    # Create executable
    add_executable(${NAME} ${SRC_FILE})
    
    # Set CUDA properties
    set_target_properties(${NAME} PROPERTIES
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
    
    # Add CUTLASS include directories
    target_include_directories(${NAME} PRIVATE 
        ${CUTLASS_DIR}/include
        ${CUTLASS_DIR}/examples/common
    )
    
    # Set NVCC flags
    target_compile_options(${NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:${COMMON_NVCC_FLAGS}>)
    
    # Link against CUTLASS if needed
    if(TARGET cutlass)
        target_link_libraries(${NAME} PRIVATE cutlass)
    endif()
endfunction()

# Add executables for each module
add_module(module1 ${CMAKE_CURRENT_SOURCE_DIR}/module1-Layouts and Tensors)
add_module(module2 ${CMAKE_CURRENT_SOURCE_DIR}/module2-Tiled Copy)
add_module(module3 ${CMAKE_CURRENT_SOURCE_DIR}/module3-Tiled MMA)
add_module(module4 ${CMAKE_CURRENT_SOURCE_DIR}/module4-Fused Bias-Add)
add_module(module5 ${CMAKE_CURRENT_SOURCE_DIR}/module5-Mainloop Pipelining)
add_module(module6 ${CMAKE_CURRENT_SOURCE_DIR}/module6-Fused Epilogues)

# Default target to build all modules
add_custom_target(all_modules DEPENDS module1 module2 module3 module4 module5 module6)

# Installation rules (optional)
install(TARGETS module1
    RUNTIME DESTINATION bin/modules
)