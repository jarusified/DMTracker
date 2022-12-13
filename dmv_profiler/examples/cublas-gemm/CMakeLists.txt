cmake_minimum_required(VERSION 3.16)

include_directories(common ../common)

add_executable(gemm-w-uvm ${exe_files} gemm-uvm.cu)
target_link_libraries(gemm-w-uvm Common -lcublas cupti DMV_LIBRARY cupti stdc++ stdc++fs nvidia-ml)
target_include_directories(gemm-w-uvm PRIVATE ${DMV_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})

add_executable(gemm-opt ${exe_files} gemm-opt.cu)
target_link_libraries(gemm-opt Common DMV_LIBRARY cupti stdc++ stdc++fs nvidia-ml)
target_include_directories(gemm-opt PRIVATE ${DMV_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
