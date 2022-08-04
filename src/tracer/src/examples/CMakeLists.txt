cmake_minimum_required(VERSION 3.11)

project(sampleProgram VERSION 0.1 LANGUAGES CXX C CUDA)

# TODO: Need to generalize the directory path.
set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/ppc64le-linux/include")
set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
SET(INCLUDE_DIRS ${PARENT_DIR}/include)
set(LIBS_SAMPLE_PROGRAM_SRC 
    kineto_playground.cpp 
    kineto_playground.cu 
    kineto_playground.cuh)

message(STATUS "Kineto is found at: ${INCLUDE_DIRS}")
message(STATUS "CUPTI is found at: ${CUPTI_INCLUDE_DIR}")
message(STATUS "Compiling: ${LIBS_SAMPLE_PROGRAM_SRC}")

add_library(kineto STATIC IMPORTED)
set_target_properties(kineto PROPERTIES
    IMPORTED_LOCATION "../libkineto.a"
)

add_executable(kineto_playground ${LIBS_SAMPLE_PROGRAM_SRC})
target_link_libraries(kineto_playground kineto cupti stdc++)
target_include_directories(kineto_playground PRIVATE ${INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
