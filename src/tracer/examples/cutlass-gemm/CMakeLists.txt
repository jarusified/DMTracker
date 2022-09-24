cmake_minimum_required(VERSION 3.16)

include_directories(common ../common)

set(CUTLASS_INCLUDE_DIR ${CUTLASS_SOURCE_DIR}/include)
message(STATUS ${CUTLASS_INCLUDE_DIR})

add_executable(cutlass-gemm gemm.cu)
target_link_libraries(cutlass-gemm KINETO_LIBRARY cupti stdc++ stdc++fs)
target_include_directories(cutlass-gemm PRIVATE ${KINETO_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR} ${CUTLASS_INCLUDE_DIR})
