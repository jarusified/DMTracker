cmake_minimum_required(VERSION 3.16)

add_executable(gemmLib ${exe_files} gemm.cu)
target_link_libraries(gemmLib Common -lcublas kineto cupti stdc++ stdc++fs)
target_include_directories(gemmLib PRIVATE ${INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
