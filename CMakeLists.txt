cmake_minimum_required(VERSION 3.20)
project(MathGPU_Kernel_Lab LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Configure CUDA architectures:
# 80: Ampere (A100)
# 86: Ampere (RTX 3090)
# 89: Ada Lovelace (RTX 4090, RTX 500 Ada)
# 90: Hopper (H100)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES "80;86;89;90")
endif()

# Compiler optimization flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O3 --use_fast_math -lineinfo --expt-relaxed-constexpr")

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/include
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/naive
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tiled
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/optimized
    ${CMAKE_CURRENT_SOURCE_DIR}/cuda/tensor_core
)

# GEMM Naive standalone executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/naive/gemm_naive.cu")
    add_executable(gemm_naive_bin
        cuda/naive/gemm_naive.cu
    )
    set_target_properties(gemm_naive_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

# GEMM Tiled standalone executable
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/cuda/tiled/gemm_tiled.cu")
    add_executable(gemm_tiled_bin
        cuda/tiled/gemm_tiled.cu
    )
    set_target_properties(gemm_tiled_bin PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()
