cmake_minimum_required(VERSION 3.16)
project(test)

# get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

set(CUDA_SOURCE_DIR "${CUDA_SOURCE_DIR}")
message(INFO " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")

set(INCLUDE_DIRS ${PARENT_DIR}/include)
set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/x86_64-linux/include")
message(STATUS ${CMAKE_CURRENT_LIST_DIR})

################################
# GTest
################################
include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

#add_library(kineto STATIC IMPORTED)
#set_target_properties(kineto PROPERTIES IMPORTED_LOCATION "../libkineto.a")

function(add_test template_name) 
add_executable(${template_name}Test ${CMAKE_CURRENT_SOURCE_DIR}/${template_name}Test.cpp ${HEADER_FILES})

target_link_libraries(${template_name}Test PRIVATE gtest gtest_main gmock)
include_directories(dmv_profiler PRIVATE ${LIBDMV_INCLUDE_DIR} ${CUPTI_INCLUDE_DIR} ${FMT_INCLUDE_DIR} ${GOOGLETEST_SOURCE_DIR}/include)
# target_include_directories(${template_name}Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR}) 

install(TARGETS ${template_name}Test DESTINATION bin) 

endfunction()


# add_test(Config)
# add_test(CuptiActivityProfiler)
add_test(CuptiCallbackApi)
#add_cuda_test(CuptiProfilerApi)
#add_test(CuptiRangeProfilerApi)