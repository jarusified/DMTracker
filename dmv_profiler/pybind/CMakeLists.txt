cmake_minimum_required(VERSION 3.16)
project(pybind)

add_subdirectory(pybind11)
pybind11_add_module(dmv_profiler "main.cpp")
target_link_libraries(dmv_profiler PRIVATE base ${CUDA_cupti_LIBRARY})
target_include_directories(dmv_profiler PRIVATE ${LIBKINETO_PUBLIC_HEADERS} ${CUPTI_INCLUDE_DIR})
set(message, ${CUDA_cupti_LIBRARY})
