cmake_minimum_required(VERSION 3.16)

include_directories(common ../common)

add_executable(cublas-gemm ${exe_files} gemm.cu)
target_link_libraries(cublas-gemm Common -lcublas KINETO_LIBRARY ${CUPTI_LIBRARY} stdc++ stdc++fs)
target_include_directories(cublas-gemm PRIVATE ${KINETO_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
