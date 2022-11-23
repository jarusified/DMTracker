cmake_minimum_required(VERSION 3.16)

include_directories(common ../common)

# add_executable(cublas-gemm ${exe_files} gemm.cu)
# target_link_libraries(cublas-gemm Common -lcublas DMV_LIBRARY cupti stdc++ stdc++fs)
# target_include_directories(cublas-gemm PRIVATE ${DMV_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR})

add_executable(basic-gemm ${exe_files} basic_gemm.cu)
target_link_libraries(basic-gemm Common DMV_LIBRARY ${CUPTI_LIBRARY} stdc++ stdc++fs nvidia-ml)
target_include_directories(basic-gemm PRIVATE ${DMV_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
