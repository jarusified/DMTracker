cmake_minimum_required(VERSION 3.16)

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

project(use-case VERSION 0.1 LANGUAGES CXX C CUDA)

# include(FetchContent)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)
set(CMAKE_CXX_COMPILE_OPTIONS_PIE "-fPIC -g")

#FetchContent_Declare(fmt
#  GIT_REPOSITORY https://github.com/fmtlib/fmt.git
#  GIT_TAG master
#)
# FetchContent_MakeAvailable(fmt)

set(exe_files
    ${CMAKE_CURRENT_LIST_DIR}/common/benchmark.cu
)

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "FMT include dir: ${FMT_INCLUDE_DIR}")

# TODO: Need to generalize the directory path for all os.
set(TARGET_NAME "ppc64le-linux")
# set(TARGET_NAME "x86_64-linux")

set(CUDA_TARGETS_INCLUDE_DIR "${CUDA_SOURCE_DIR}targets/${TARGET_NAME}/include")
set(CUDA_TARGETS_LIB_DIR "${CUDA_SOURCE_DIR}targets/${TARGET_NAME}/lib")
message(STATUS "CUPTI include dir: ${CUDA_TARGETS_INCLUDE_DIR}")

find_library(CUPTI_LIBRARY cupti HINTS "${CUDA_TARGETS_LIB_DIR}")
find_library(CUDA_LIBRARY cudart HINTS "${CUDA_TARGETS_LIB_DIR}")

message(STATUS "DMV source dir: ${DMV_SOURCE_DIR}")

set(DMV_LIB_PATH "${DMV_SOURCE_DIR}/${CMAKE_STATIC_LIBRARY_PREFIX}dmv${CMAKE_STATIC_LIBRARY_SUFFIX}")
set(DMV_INCLUDE_DIR ${DMV_SOURCE_DIR}/include)

message(STATUS "DMV source dir: ${DMV_SOURCE_DIR}")
message(STATUS "DMV include dir: ${DMV_INCLUDE_DIR}")
message(STATUS "DMV lib path: ${DMV_LIB_PATH}")

add_library(DMV_LIBRARY STATIC IMPORTED)
set_target_properties(DMV_LIBRARY PROPERTIES 
  IMPORTED_LOCATION "${DMV_LIB_PATH}")

message(STATUS ${DMV_LIBRARY})
if(NOT ${DMV_LIBRARY})
  message(FATAL_ERROR "DMV library not found")
endif()


add_subdirectory(common)
add_subdirectory(sqrt)
# add_subdirectory(distributed-volume-path-tracing)
# add_subdirectory(cutlass-gemm)
# add_subdirectory(marbl-matprops-miniapp)
# add_subdirectory(cublas-gemm)
