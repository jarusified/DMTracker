cmake_minimum_required(VERSION 3.16)

project(use-case LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(exe_files
    ${CMAKE_CURRENT_LIST_DIR}/common/benchmark.cu
)

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "FMT include dir: ${FMT_INCLUDE_DIR}")

# TODO: Need to generalize the directory path for all os.
set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/x86_64-linux/include")
message(STATUS "CUPTI include dir: ${CUPTI_INCLUDE_DIR}")

message(STATUS "Kineto source dir: ${KINETO_SOURCE_DIR}")

set(KINETO_INCLUDE_DIR ${KINETO_SOURCE_DIR}/include)
message(STATUS "Kineto include dir: ${KINETO_INCLUDE_DIR}")

set(KINETO_LIB_PATH "${KINETO_SOURCE_DIR}/libkineto.a")
message(STATUS ${CMAKE_STATIC_LIBRARY_PREFIX}kineto${CMAKE_STATIC_LIBRARY_SUFFIX})
message(STATUS ${KINETO_SOURCE_DIR})

add_library(KINETO_LIBRARY STATIC IMPORTED)
set_target_properties(KINETO_LIBRARY PROPERTIES 
  IMPORTED_LOCATION "${KINETO_SOURCE_DIR}/libkineto.a")

if(NOT ${KINETO_LIBRARY})
  message(FATAL_ERROR "Kineto library not found")
endif()



add_subdirectory(common)
add_subdirectory(sqrt)
add_subdirectory(cublas-gemm)
add_subdirectory(cutlass-gemm)
