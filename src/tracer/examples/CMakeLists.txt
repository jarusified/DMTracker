cmake_minimum_required(VERSION 3.16)


project(use-case LANGUAGES CXX CUDA)

set(exe_files
    ${CMAKE_CURRENT_LIST_DIR}/common/benchmark.cu
)

add_subdirectory(common)
add_subdirectory(sqrt)
add_subdirectory(cublas-gemm)
