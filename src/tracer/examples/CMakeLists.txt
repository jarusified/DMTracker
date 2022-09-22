cmake_minimum_required(VERSION 3.16)

project(use-case LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(exe_files
    ${CMAKE_CURRENT_LIST_DIR}/common/benchmark.cu
)

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "FMT include dir: ${FMT_INCLUDE_DIR}")

# TODO: Need to generalize the directory path.
set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/ppc64le-linux/include")
message(STATUS "CUPTI include dir: ${CUPTI_INCLUDE_DIR}")

set(KINETO_INCLUDE_DIR ${KINETO_SOURCE_DIR}/include)
message(STATUS "Kineto include dir: ${KINETO_INCLUDE_DIR}")

set(KINETO_LIB_PATH "${KINETO_SOURCE_DIR}/libkineto.a")
message(STATUS ${CMAKE_STATIC_LIBRARY_PREFIX}kineto${CMAKE_STATIC_LIBRARY_SUFFIX})
message(STATUS ${KINETO_SOURCE_DIR})

find_library(KINETO_LIBRARY
  NAMES ${CMAKE_STATIC_LIBRARY_PREFIX}kineto${CMAKE_STATIC_LIBRARY_SUFFIX}
  PATHS KINETO_SOURCE_DIR)
if(NOT KINETO_LIBRARY)
  message(FATAL_ERROR "Kineto library not found")
endif()
# set_target_properties(kineto PROPERTIES
#    IMPORTED_LOCATION "../libkineto.a"
#)


add_subdirectory(common)
add_subdirectory(sqrt)
add_subdirectory(cublas-gemm)
