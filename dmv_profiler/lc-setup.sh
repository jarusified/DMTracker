#!/usr/bin/env bash

# Load required module/version
module load spectrum-mpi/rolling-release
module load cuda/11.1.0
module load cmake/3.16.8
module load gcc/8.3.1

# Create a Python env for the project.
if [ ! -d "env" ]; then
  echo "Virtualenv `env` not created. Creating one!!"
  virtualenv env
fi

source ./env/bin/activate

# Activate spack
# NOTE: this only works locally.
. /g/g91/kesavan/spack/share/spack/setup-env.sh
export PATH=/g/g91/kesavan/spack/bin:$PATH

spack install fmt googletest@1.8.1

spack load fmt googletest/gdlwgea

CUDA_SOURCE_DIR=$(which nvcc | cut -d'/' -f-6)
echo "FOUND CUDA: " ${CUDA_SOURCE_DIR}

export CUDA_SOURCE_DIR=${CUDA_SOURCE_DIR}
export FMT_SOURCE_DIR=`spack location -i fmt`
export GOOGLETEST_SOURCE_DIR=`spack location -i googletest`
# export PYBIND_SOURCE_DIR=`spack location -i py-pybind11`

echo "CUDA_SOURCE_DIR        = $CUDA_SOURCE_DIR"
echo "FMT_SOURCE_DIR         = $FMT_SOURCE_DIR"
echo "GOOGLETEST_SOURCE_DIR  = $GOOGLETEST_SOURCE_DIR"
# echo "PYBIND_SOURCE_DIR      = $PYBIND_SOURCE_DIR"
