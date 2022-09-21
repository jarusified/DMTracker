cmake_minimum_required(VERSION 3.16)

project(sampleProgram VERSION 0.1 LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)

# TODO: Need to generalize the directory path.
set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/x86_64-linux/include")
message(STATUS "CUPTI is found at: ${CUPTI_INCLUDE_DIR}")

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")

# get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
set(INCLUDE_DIRS ${KINETO_SOURCE_DIR}/include)
message(STATUS "Kineto is found at: ${INCLUDE_DIRS}")

set(LIBS_SAMPLE_PROGRAM_SRC 
    main.cpp 
    utils.cu 
    main.cuh)
message(STATUS "Compiling: ${LIBS_SAMPLE_PROGRAM_SRC}")

add_library(kineto STATIC IMPORTED)
set(KINETO_LIB_PATH "${KINETO_SOURCE_DIR}/libkineto.a")
set_target_properties(kineto PROPERTIES
    IMPORTED_LOCATION "../libkineto.a"
)
message(STATUS "KINETO: ${KINETO_LIB_PATH} ${kineto}")


add_executable(main ${LIBS_SAMPLE_PROGRAM_SRC})
target_link_libraries(main kineto cupti stdc++ stdc++fs)
target_include_directories(main PRIVATE ${INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
