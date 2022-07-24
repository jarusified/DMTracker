cmake_minimum_required(VERSION 3.11)

project(sampleProgram VERSION 0.1 LANGUAGES CXX C CUDA)

function(find_static_library LIB_NAME OUT)
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".so")
    find_library(FOUND_PROFILER_STATIC ${LIB_NAME} HINTS "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/tracer")

    if (FOUND_PROFILER_STATIC)
        message(SEND_ERROR "Found library ${LIB_NAME}: ${FOUND_PROFILER_STATIC}")
        # get_filename_component(ABS_FILE ${FOUND_${LIB_NAME}_STATIC} ABSOLUTE)
    else()
        message(SEND_ERROR "Unable to find library ${LIB_NAME}")
    endif()

    set(${OUT} ${FOUND_PROFILER_STATIC})
endfunction()


SET(INCLUDE_DIRS "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/tracer/include")
SET(HEADER_FILES ${INCLUDE_DIRS}/libkineto.h)
set(LIBS_SAMPLE_PROGRAM_SRC kineto_playground.cpp kineto_playground.cu kineto_playground.cuh ${HEADER_FILES})

message(STATUS ${INCLUDE_DIRS})
message(STATUS ${LIBS_SAMPLE_PROGRAM_SRC})

add_library(kineto SHARED IMPORTED)
set_target_properties(kineto PROPERTIES
    IMPORTED_LOCATION "/home/suraj/Work/llnl/nvidia-data-movement-experiments/src/tracer/libkineto.a"
)

add_executable(kineto_playground ${LIBS_SAMPLE_PROGRAM_SRC})

include_directories(kineto_playground ${INCLUDE_DIRS})
target_link_libraries(kineto_playground kineto)