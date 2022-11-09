#!/usr/bin/env bash

# Clean up old build, if it exists.
rm -rf CMakeFiles CMakeCache.txt compile_commands.json cmake_install.cmake Makefile

cmake . -DCUDA_SOURCE_DIR=$CUDA_SOURCE_DIR -DFMT_SOURCE_DIR=$FMT_SOURCE_DIR -DDMV_SOURCE_DIR=$DMV_SOURCE_DIR

cmake --build . --parallel 16