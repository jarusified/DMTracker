#!/usr/bin/env bash
set -e 

module load cuda/11.7.0
module load gcc/8.3.1
module load cmake/3.18.0

export CC=`which gcc`
export CXX=`which g++`

export CUDA_SOURCE_DIR=/usr/tce/packages/cuda/cuda-11.7.0
export FMT_SOURCE_DIR=/g/g91/kesavan/nvidia-data-movement-experiments/softwares/fmt
export GOOGLETEST_SOURCE_DIR=/g/g91/kesavan/nvidia-data-movement-experiments/softwares/googletest

cmake . -DFMT_SOURCE_DIR=$FMT_SOURCE_DIR -DGOOGLETEST_SOURCE_DIR=$GOOGLETEST_SOURCE_DIR -DCUDA_SOURCE_DIR=$CUDA_SOURCE_DIR

make

