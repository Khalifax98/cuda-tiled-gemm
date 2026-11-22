cmake_minimum_required(VERSION 3.18)
project(cuda-tiled-gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(CUDAToolkit REQUIRED)

add_executable(gemm_benchmark
    src/main.cu
    src/naive_gemm.cu
    src/tiled_gemm.cu
    src/coarsened_gemm.cu
)

target_include_directories(gemm_benchmark PRIVATE include)

target_compile_options(gemm_benchmark PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3 -lineinfo --use_fast_math>
    $<$<COMPILE_LANGUAGE:CXX>:-O3>
)

target_link_libraries(gemm_benchmark PRIVATE CUDA::cudart CUDA::cublas)
