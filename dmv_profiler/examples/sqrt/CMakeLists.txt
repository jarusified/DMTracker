cmake_minimum_required(VERSION 3.16)

set(LIBS_SAMPLE_PROGRAM_SRC 
    main.cpp 
    utils.cu 
    main.cuh)
message(STATUS "Compiling: ${LIBS_SAMPLE_PROGRAM_SRC}")

add_executable(sqrt ${LIBS_SAMPLE_PROGRAM_SRC})
target_link_libraries(sqrt DMV_LIBRARY cupti stdc++ stdc++fs -lnvidia-ml)
target_include_directories(sqrt PRIVATE ${DMV_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR})
