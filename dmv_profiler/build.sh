#!/usr/bin/env bash

# TODO (surajk): Use this for python installation.

# Clean up old build, if it exists.
rm -rf CMakeFiles CMakeCache.txt compile_commands.json cmake_install.cmake Makefile libdmv.a

# Load required module/version
module load spectrum-mpi/rolling-release
module load cmake/3.16.8
module load gcc/8.3.1
module load python/3.8.2

# Force the compiler to pick the correct versions.
export CC=`which gcc`
export CXX=`which g++`

cmake . -DCUDA_SOURCE_DIR=$CUDA_SOURCE_DIR -DFMT_SOURCE_DIR=$FMT_SOURCE_DIR -DGOOGLETEST_SOURCE_DIR=$GOOGLETEST_SOURCE_DIR

export DMV_SOURCE_DIR=$(pwd)
echo "DMV_SOURCE_DIR        = $DMV_SOURCE_DIR"

cmake --build .