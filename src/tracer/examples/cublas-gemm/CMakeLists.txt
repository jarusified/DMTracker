cmake_minimum_required(VERSION 3.16)

include_directories(common ../common)

add_executable(gemmLib ${exe_files} gemm.cu)
target_link_libraries(gemmLib Common -lcublas ${KINETO_LIBRARY} cupti stdc++ stdc++fs)
target_include_directories(gemmLib PRIVATE ${KINETO_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
