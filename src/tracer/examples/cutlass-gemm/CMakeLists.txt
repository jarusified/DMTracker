cmake_minimum_required(VERSION 3.16)

include_directories(common ../common)

set(CUTLASS_INCLUDE_DIR ${KINETO_SOURCE_DIR}/include)

add_executable(cutlass-gemm ${exe_files} gemm.cu)
target_link_libraries(cutlass-gemm Common -lcutlass ${KINETO_LIBRARY} cupti stdc++ stdc++fs)
target_include_directories(cutlass-gemm PRIVATE ${KINETO_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
