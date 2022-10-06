cmake_minimum_required(VERSION 3.16)
project(binding)

add_subdirectory(pybind11)
pybind11_add_module(dmv_profiler "main.cpp")
target_link_directories(dmv_profiler PUBLIC "${CUDA_LIB_DIR}")
include_directories(dmv_profiler PRIVATE ${LIBKINETO_PUBLIC_HEADERS} ${CUPTI_INCLUDE_DIR})

find_library(CUPTI_LIBRARY cupti HINTS "${CUPTI_LIB_DIR}")
find_library(CUDA_LIBRARY cudart HINTS "${CUPTI_LIB_DIR}")
target_link_libraries(dmv_profiler PUBLIC base ${CUPTI_LIBRARY} ${CUDA_LIBRARY})
set_target_properties(dmv_profiler PROPERTIES POSITION_INDEPENDENT_CODE TRUE)
