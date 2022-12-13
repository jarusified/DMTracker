#!/usr/bin/env bash

CUDA_VERSION=11.1.0
GCC_VERSION=8.3.1
PYTHON_VERSION=3.8.2
CMAKE_VERSiON=3.20.2

# Load packages
module load cuda/${CUDA_VERSION}
module load python/${PYTHON_VERSION}
module load gcc/${GCC_VERSION}
module load cmake/${CMAKE_VERSiON}
module load spectrum-mpi/rolling-release

# Create a Python env for the project.
if [ ! -d "env" ]; then
  echo "Virtualenv `env` not created. Creating one!!"
  python3 -m virtualenv env
fi

source ./env/bin/activate

# Activate spack
# NOTE: this only works locally.
. /g/g91/kesavan/spack/share/spack/setup-env.sh
export PATH=/g/g91/kesavan/spack/bin:$PATH

# Install fmt and googletest -> dependencies.
# TODO (surajk): Do not install if already installed.
# spack install fmt@9.0.0 googletest@1.8.1 py-pybind11
spack load fmt googletest/gdlwgea #py-pybind11/lwmygkz

# Find CUDA source path
CUDA_SOURCE_DIR=$(which nvcc | cut -d'/' -f-6)
echo "FOUND CUDA: " ${CUDA_SOURCE_DIR}

# Export variables
# TODO (surajk): Hard coded here because module load is inconsistent!
export CUDA_SOURCE_DIR=/usr/tce/packages/cuda/cuda-${CUDA_VERSION}
export FMT_SOURCE_DIR=`spack location -i fmt`
export GOOGLETEST_SOURCE_DIR=`spack location -i googletest`
# export PYBIND_SOURCE_DIR=`spack location -i py-pybind11`

# For debugging.
echo "CUDA_SOURCE_DIR        = $CUDA_SOURCE_DIR"
echo "FMT_SOURCE_DIR         = $FMT_SOURCE_DIR"
echo "GOOGLETEST_SOURCE_DIR  = $GOOGLETEST_SOURCE_DIR"
# echo "PYBIND_SOURCE_DIR      = $PYBIND_SOURCE_DIR"
