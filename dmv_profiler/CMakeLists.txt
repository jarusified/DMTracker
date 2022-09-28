cmake_minimum_required(VERSION 3.16)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/modules")

#install libraries into correct locations on all platforms
include(GNUInstallDirs)

# function to extract filelists from libkineto_defs.bzl file
find_package(PythonInterp)
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

set(KINETO_LIBRARY_TYPE "static" CACHE STRING "Type of library (default, static or shared) to build")
set_property(CACHE KINETO_LIBRARY_TYPE PROPERTY STRINGS default shared)

set(LIBKINETO_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/lib")
set(LIBKINETO_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include")
set(LIBKINETO_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/bin")
set(LIBKINETO_THIRDPARTY_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(KINETO_BUILD_TESTS "Build kineto unit tests" ON)
option(KINETO_BUILD_SAMPLE_PROGRAMS "Build kineto sample programs" ON)

# We should default to a Release build
if (NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "")
    set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

if (NOT CUDA_SOURCE_DIR)
    set(CUDA_SOURCE_DIR "$ENV{CUDA_SOURCE_DIR}")
    message(INFO " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")
endif()

# Set LIBKINETO_NOCUPTI to explicitly disable CUPTI
# Otherwise, CUPTI is disabled if not found
IF (NOT CUDA_SOURCE_DIR OR NOT CUPTI_INCLUDE_DIR OR NOT CUDA_cupti_LIBRARY)
    set(LIBKINETO_NOCUPTI ON CACHE BOOL "" FORCE)
endif()

# Define file lists
if (NOT LIBKINETO_NOCUPTI)
    get_filelist("get_libkineto_cpu_only_srcs(with_api=False)" LIBKINETO_SRCS)
    message(INFO " CUPTI unavailable or disabled - not building GPU profilers")
else()
    get_filelist("get_libkineto_cupti_srcs(with_api=True)" LIBKINETO_SRCS)
endif()
get_filelist("get_libkineto_public_headers()" LIBKINETO_PUBLIC_HEADERS)
get_filelist("get_libkineto_api_srcs()" LIBKINETO_API_SRCS)

add_library(base OBJECT ${LIBKINETO_SRCS})
add_library(api OBJECT ${LIBKINETO_API_SRCS})

# Make libraries depend on libkineto_defs.bzl
# add_custom_target(libkineto_defs.bzl DEPENDS libkineto_defs.bzl)
# add_dependencies(base libkineto_defs.bzl)

set_target_properties(base api PROPERTIES
      CXX_STANDARD 14
      CXX_STANDARD_REQUIRED YES
      CXX_EXTENSIONS NO)

set(KINETO_COMPILE_OPTIONS "-DKINETO_NAMESPACE=libkineto")
list(APPEND KINETO_COMPILE_OPTIONS "-DFMT_HEADER_ONLY")
list(APPEND KINETO_COMPILE_OPTIONS "-std=c++14")
if (LIBKINETO_NOCUPTI)
    list(APPEND KINETO_COMPILE_OPTIONS "-DHAS_CUPTI")
endif()
message(STATUS "CUPTI=${LIBKINETO_NOCUPTI}")

target_compile_options(base PRIVATE "${KINETO_COMPILE_OPTIONS}")
target_compile_options(api PRIVATE "${KINETO_COMPILE_OPTIONS}")

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")
message(STATUS "Kineto: FMT_SOURCE_DIR = ${FMT_SOURCE_DIR}")
message(STATUS "Kineto: FMT_INCLUDE_DIR = ${FMT_INCLUDE_DIR}")

if (NOT CUPTI_INCLUDE_DIR)
    set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/x86_64-linux/include")
    message(STATUS "Kineto: CUPTI_INCLUDE_DIR = ${CUPTI_INCLUDE_DIR}")
endif()
if (NOT CUDA_INCLUDE_DIRS)
    set(CUDA_INCLUDE_DIRS "${CUDA_SOURCE_DIR}/include")
endif()

set(PYBIND_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/pybind11/include")

target_include_directories(base PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${LIBKINETO_SOURCE_DIR}>
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${PYBIND_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUPTI_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${CUDA_INCLUDE_DIRS}>)

target_include_directories(api PUBLIC
      $<BUILD_INTERFACE:${LIBKINETO_INCLUDE_DIR}>
      $<BUILD_INTERFACE:${FMT_INCLUDE_DIR}>
)

if(KINETO_LIBRARY_TYPE STREQUAL "default")
  add_library(kineto $<TARGET_OBJECTS:base> $<TARGET_OBJECTS:api>)
elseif(KINETO_LIBRARY_TYPE STREQUAL "static")
  add_library(kineto STATIC
    $<TARGET_OBJECTS:base>
    $<TARGET_OBJECTS:api>)
elseif(KINETO_LIBRARY_TYPE STREQUAL "shared")
  add_library(kineto SHARED
    $<TARGET_OBJECTS:base>)
  set_property(TARGET base PROPERTY POSITION_INDEPENDENT_CODE ON)
  set_target_properties(kineto PROPERTIES
    CXX_VISIBILITY_PRESET hidden)
else()
  message(FATAL_ERROR "Unsupported library type ${KINETO_LIBRARY_TYPE}")
endif()

if(NOT LIBKINETO_NOCUPTI)
  target_link_libraries(kineto "${CUDA_cupti_LIBRARY}")
endif()

add_subdirectory(pybind11)
pybind11_add_module(dmv_profiler "lib/pybind.cpp")
target_link_libraries(dmv_profiler PRIVATE kineto)

install(TARGETS kineto EXPORT kinetoLibraryConfig
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})

install(FILES ${LIBKINETO_PUBLIC_HEADERS}
  DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}/kineto")

install(EXPORT kinetoLibraryConfig DESTINATION share/cmake/kineto
  FILE kinetoLibraryConfig.cmake)
