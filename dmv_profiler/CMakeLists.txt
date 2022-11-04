cmake_minimum_required(VERSION 3.16)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")
set(CMAKE_CXX_FLAGS "-fPIC")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from main.bzl file
find_package (PythonInterp 3.6)
function(get_filelist name outputvar)
  execute_process(
    COMMAND "${PYTHON_EXECUTABLE}" -c
            "exec(open('main.bzl').read());print(';'.join(${name}))"
    WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
    OUTPUT_VARIABLE _tempvar)
  string(REPLACE "\n" "" _tempvar "${_tempvar}")
  set(${outputvar} ${_tempvar} PARENT_SCOPE)
endfunction()

project(dmv_profiler VERSION 0.1 LANGUAGES CXX C CUDA)

message("Compiler Version: ${CMAKE_CXX_COMPILER_VERSION}")

set(DMV_LIBRARY_TYPE "static" CACHE STRING "Type of library (default, static or shared) to build")
set_property(CACHE DMV_LIBRARY_TYPE PROPERTY STRINGS default shared)

set(LIBDMV_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(LIBDMV_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIBDMV_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(LIBDMV_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(DMV_BUILD_TESTS "Build dmv unit tests" ON)
option(DMV_BUILD_SAMPLE_PROGRAMS "Build dmv sample programs" ON)

# We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if (NOT CUDA_SOURCE_DIR)
    set(CUDA_SOURCE_DIR "$ENV{CUDA_SOURCE_DIR}")
    message(INFO " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")
endif()

# Set LIBDMV_NOCUPTI to explicitly disable CUPTI
# Otherwise, CUPTI is disabled if not found
IF (NOT CUDA_SOURCE_DIR OR NOT CUPTI_INCLUDE_DIR OR NOT CUDA_cupti_LIBRARY)
    set(LIBDMV_NOCUPTI ON CACHE BOOL "" FORCE)
endif()

# Define file lists
get_filelist("get_libdmv_cupti_srcs(with_api=True)" LIBDMV_SRCS)
get_filelist("get_libdmv_public_headers()" LIBDMV_PUBLIC_HEADERS)

add_library(base OBJECT ${LIBDMV_SRCS})

set_target_properties(base PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS YES)

set(DMV_COMPILE_OPTIONS "-DDMV_NAMESPACE=libdmv")
list(APPEND DMV_COMPILE_OPTIONS "-DFMT_HEADER_ONLY")
list(APPEND DMV_COMPILE_OPTIONS "-std=c++14")

if (LIBDMV_NOCUPTI)
    list(APPEND DMV_COMPILE_OPTIONS "-DHAS_CUPTI")
endif()
message(STATUS "CUPTI=${LIBDMV_NOCUPTI}")

target_compile_options(base PRIVATE "${DMV_COMPILE_OPTIONS}")

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "DMV: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
message(STATUS "DMV: FMT_INCLUDE_DIR = ${FMT_INCLUDE_DIR}")

set(TARGET_NAME "ppc64le-linux")
# set(TARGET_NAME "x86_64-linux")

if (NOT CUPTI_INCLUDE_DIR)
    set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/${TARGET_NAME}/include")
    set(CUPTI_LIB_DIR "${CUDA_SOURCE_DIR}/targets/${TARGET_NAME}/lib")
    message(STATUS "DMV: CUPTI_INCLUDE_DIR = ${CUPTI_INCLUDE_DIR}")
    message(STATUS "DMV: CUPTI_LIB_DIR = ${CUPTI_LIB_DIR}")
endif()
if (NOT CUDA_INCLUDE_DIRS)
    set(CUDA_INCLUDE_DIRS "${CUDA_SOURCE_DIR}/include")
endif()

target_include_directories(base PUBLIC
      $<BUILD_INTERFACE:${LIBDMV_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBDMV_SOURCE_DIR}>
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUPTI_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

add_subdirectory(pybind)

if(DMV_LIBRARY_TYPE STREQUAL "default")
  add_library(dmv $<TARGET_OBJECTS:base>)
elseif(DMV_LIBRARY_TYPE STREQUAL "static")
  add_library(dmv STATIC $<TARGET_OBJECTS:base>)
elseif(DMV_LIBRARY_TYPE STREQUAL "shared")
  add_library(dmv SHARED $<TARGET_OBJECTS:base>)
  set_property(TARGET base PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(dmv PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
else()
  message(FATAL_ERROR "Unsupported library type ${DMV_LIBRARY_TYPE}")
endif()

if(NOT LIBDMV_NOCUPTI)
  target_link_libraries(dmv cupti)
endif()


install(TARGETS dmv EXPORT dmvLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${LIBDMV_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/dmv")

install(EXPORT dmvLibraryConfig DESTINATION share/cmake/dmv
  FILE dmvLibraryConfig.cmake)

# add_subdirectory(test)
