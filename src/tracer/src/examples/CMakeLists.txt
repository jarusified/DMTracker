cmake_minimum_required(VERSION 3.11)

project(sampleProgram VERSION 0.1 LANGUAGES CXX C CUDA)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
SET(INCLUDE_DIRS ${PARENT_DIR}/include)
SET(HEADER_FILES ${INCLUDE_DIRS}/libkineto.h)
set(LIBS_SAMPLE_PROGRAM_SRC kineto_playground.cpp kineto_playground.cu kineto_playground.cuh ${HEADER_FILES})

message(STATUS ${INCLUDE_DIRS})
message(STATUS ${LIBS_SAMPLE_PROGRAM_SRC})

add_library(kineto STATIC IMPORTED)
set_target_properties(kineto PROPERTIES
    IMPORTED_LOCATION "../libkineto.a"
)

add_executable(kineto_playground ${LIBS_SAMPLE_PROGRAM_SRC})
target_link_libraries(kineto_playground kineto)
target_include_directories(kineto_playground PRIVATE ${INCLUDE_DIRS})
