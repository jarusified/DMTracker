#!/bin/zsh
# Create a Python env for the project.
if [ ! -d "env" ]; then
  echo "Virtualenv `env` not created. Creating one!!"
  virtualenv env
fi

source env/bin/activate

# if ! [ -x "$(command -v spack)" ]; then
#   echo 'Error: spack is not installed. Please refer https://spack-tutorial.readthedocs.io/en/latest/tutorial_basics.html for installation.' >&2
# fi

spack install fmt googletest py-pybind11

spack load fmt googletest py-pybind11

# TODO (surajk): Add pybind as a git-submodule.

export CUDA_SOURCE_DIR=/usr/local/cuda-11.7
export FMT_SOURCE_DIR=`spack location -i fmt`
export GOOGLETEST_SOURCE_DIR=`spack location -i googletest`
export PYBIND_SOURCE_DIR=`spack location -i py-pybind11`

echo "CUDA_SOURCE_DIR        = $CUDA_SOURCE_DIR"
echo "FMT_SOURCE_DIR         = $FMT_SOURCE_DIR"
echo "GOOGLETEST_SOURCE_DIR  = $GOOGLETEST_SOURCE_DIR"
echo "PYBIND_SOURCE_DIR      = $PYBIND_SOURCE_DIR"