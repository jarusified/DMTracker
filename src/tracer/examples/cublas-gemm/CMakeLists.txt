cmake_minimum_required(VERSION 3.16)

project(gemmProgram VERSION 0.1 LANGUAGES CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)

# TODO: Need to generalize the directory path.
set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/x86_64-linux/include")
message(STATUS "CUPTI is found at: ${CUPTI_INCLUDE_DIR}")

set(FMT_INCLUDE_DIR "${FMT_SOURCE_DIR}/include")

include_directories(common ../common)

set(INCLUDE_DIRS ${KINETO_SOURCE_DIR}/include)
message(STATUS "Kineto is found at: ${INCLUDE_DIRS}")


add_library(kineto STATIC IMPORTED)
set(KINETO_LIB_PATH "${KINETO_SOURCE_DIR}/libkineto.a")
set_target_properties(kineto PROPERTIES
    IMPORTED_LOCATION "../libkineto.a"
)
message(STATUS "KINETO: ${KINETO_LIB_PATH} ${kineto}")

add_executable(gemmLib ${exe_files} gemm.cu)
target_link_libraries(gemmLib Common -lcublas kineto cupti stdc++ stdc++fs)
target_include_directories(gemmLib PRIVATE ${INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
