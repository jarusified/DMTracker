SET(GTEST_UTILS_SRC utils.cpp)

get_filename_component(PARENT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)

SET(INCLUDE_DIRS ${PARENT_DIR}/include)
message(STATUS ${CMAKE_CURRENT_LIST_DIR})

add_library(kinetox STATIC IMPORTED)
set_target_properties(kinetox PROPERTIES IMPORTED_LOCATION "${PARENT_DIR}/libkineto.a")

function(add_test template_name)
    add_executable(${template_name}Test ${CMAKE_CURRENT_SOURCE_DIR}/${template_name}Test.cpp ${HEADER_FILES})
    target_link_libraries(${template_name}Test PRIVATE gtest_main kinetox)
    target_include_directories(${template_name}Test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR} ${INCLUDE_DIRS})
    if (${DEV_BUILD})
        install(TARGETS ${template_name}Test DESTINATION bin) 
    endif()
endfunction() 

add_test(Config)
# add_test(CuptiActivityProfiler)
# add_test(CuptiCallbackApi)
# add_cuda_test(CuptiProfilerApi)
# add_test(CuptiRangeProfilerApi)