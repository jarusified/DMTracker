cmake_minimum_required(VERSION 3.16)

set(common_source_files
    ResultDatabase.cpp
    ResultDatabase.h
    OptionParser.cpp
    OptionParser.h
    Option.cpp
    Option.h)

add_library(Common STATIC ${common_source_files})