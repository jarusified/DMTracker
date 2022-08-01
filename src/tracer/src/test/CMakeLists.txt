get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

if (NOT CUDA_SOURCE_DIR)
    set(CUDA_SOURCE_DIR "$ENV{CUDA_SOURCE_DIR}")
    message(INFO " CUDA_SOURCE_DIR = ${CUDA_SOURCE_DIR}")
endif()

SET(INCLUDE_DIRS ${PARENT_DIR}/include)
set(CUPTI_INCLUDE_DIR "${CUDA_SOURCE_DIR}/targets/x86_64-linux/include")
message(STATUS ${CMAKE_CURRENT_LIST_DIR})

# add_library(kineto STATIC IMPORTED)
# set_target_properties(kineto PROPERTIES IMPORTED_LOCATION "../libkineto.a")

function(add_test template_name)
    add_executable(${template_name}Test ${CMAKE_CURRENT_SOURCE_DIR}/${template_name}Test.cpp ${HEADER_FILES})
    target_link_libraries(${template_name}Test PRIVATE gtest_main kineto)
    target_include_directories(${template_name}Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS} ${CUPTI_INCLUDE_DIR})
    if (${DEV_BUILD})
        install(TARGETS ${template_name}Test DESTINATION bin) 
    endif()
endfunction() 

add_test(Config)
# add_test(CuptiActivityProfiler)
# add_test(CuptiCallbackApi)
# add_cuda_test(CuptiProfilerApi)
# add_test(CuptiRangeProfilerApi)