cmake_minimum_required(VERSION 3.16)

project(use-case LANGUAGES CXX CUDA)

option(WITH_CALIPER "Build with Caliper" TRUE)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")
set(CMAKE_CUDA_COMPILER "/home/suraj/Work/llnl/spack/opt/spack/linux-ubuntu20.04-skylake/gcc-9.3.0/cuda-11.1.0-5zknzvukfvdi2ihfhj25fcqj4gktqesr/bin/nvcc")

set(CMAKE_THREAD_LIBS_INIT "-lpthread")
set(CMAKE_HAVE_THREADS_LIBRARY 1)
set(CMAKE_USE_WIN32_THREADS_INIT 0)
set(CMAKE_USE_PTHREADS_INIT 1)
set(THREADS_PREFER_PTHREAD_FLAG ON)

if (WITH_CALIPER)
  find_package(caliper REQUIRED)
  if(caliper_FOUND)
    message(STATUS "Found Caliper: " ${caliper_DIR})
    #find_path(caliper_DIR caliper/cali.h
     # HINTS ${caliper_DIR}/include)
    include_directories(${caliper_DIR}/include)
    add_definitions("-DUSE_CALIPER")
    set(CALIPER_CMAKE_MSG "Yes, using Caliper")
    list(APPEND EXTERNAL_LIBS caliper)
  else()
    message(STATUS "Caliper not found!")
  endif()
endif()

add_subdirectory(src)

message(STATUS "External libs: " ${EXTERNAL_LIBS})
target_link_libraries(cublas-gemm ${EXTERNAL_LIBS})
