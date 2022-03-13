cmake_minimum_required(VERSION 3.8)


set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_C_COMPILER "/usr/bin/gcc")
set(CMAKE_CXX_COMPILER "/usr/bin/g++")

set(common_source_files
    ResultDatabase.cpp
    ResultDatabase.h
    OptionParser.cpp
    OptionParser.h
    Option.cpp
    Option.h
    ProgressBar.cpp
    ProgressBar.h)

add_library(Common STATIC ${common_source_files})
