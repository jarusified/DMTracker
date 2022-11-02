#!/usr/bin/env bash

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
spack install fmt@9.0.0 googletest@1.8.1
spack load fmt googletest/gdlwgea

# Find CUDA source path
CUDA_SOURCE_DIR=$(which nvcc | cut -d'/' -f-6)
echo "FOUND CUDA: " ${CUDA_SOURCE_DIR}

# Export variables
export CUDA_SOURCE_DIR=${CUDA_SOURCE_DIR}
export FMT_SOURCE_DIR=`spack location -i fmt`
export GOOGLETEST_SOURCE_DIR=`spack location -i googletest`

echo "CUDA_SOURCE_DIR        = $CUDA_SOURCE_DIR"
echo "FMT_SOURCE_DIR         = $FMT_SOURCE_DIR"
echo "GOOGLETEST_SOURCE_DIR  = $GOOGLETEST_SOURCE_DIR"