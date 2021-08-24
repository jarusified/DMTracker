command_exists () {
  command -v "$1" &> /dev/null 2>&1
}

SPACK_COMMAND="spack"
if command_exists ${SPACK_COMMAND}; then
  echo "Found spack in $(which $SPACK_COMMAND)."
else
  echo "Cloning $SPACK_COMMAND from Github."
  git clone https://github.com/spack/spack.git
fi

SPACK_PATH=`which $SPACK_COMMAND`
# Technical debt: Could break if spack is moved to a different folder in future.
SPACK_HOME=$( echo ${SPACK_PATH%/*/*} ) 

source ${SPACK_HOME}/share/spack/setup-env.sh

spack compiler find
spack external find
spack install boost~shared+pic dyninst@10.2.1%gcc
spack install papi
spack load -r boost dyninst arch=`spack arch`

PYTHON_VERSION=3.8

TIMEMORY_COMMAND="timem"
if command_exists ${TIMEMORY_COMMAND}; then
  echo "Found ${TIMEMORY_COMMAND} in $(which $TIMEMORY_COMMAND)."
else
  echo "Cloning ${TIMEMORY_COMMAND} from Github."
  git clone https://github.com/NERSC/timemory.git timemory-source
  python${PYTHON_VERSION} -m pip install --user -r ${PWD}/timemory-source/.requirements/runtime.txt
  python${PYTHON_VERSION} -m pip install --user -r ${PWD}/timemory-source/.requirements/mpi_runtime.txt

  cmake -B ${PWD}/timemory-build \
    -D CMAKE_INSTALL_PREFIX=timemory-install \
    -D CMAKE_BUILD_TYPE=RelWithDebInfo \
    -D TIMEMORY_USE_DYNINST=ON \
    -D TIMEMORY_USE_GOTCHA=ON \
    -D TIMEMORY_USE_PAPI=ON \
    -D TIMEMORY_USE_MPI=ON \
    -D TIMEMORY_USE_PYTHON=ON \
    -D TIMEMORY_BUILD_TOOLS=ON \
    -D PYTHON_EXECUTABLE=$(which python${PYTHON_VERSION}) \
    timemory-source
  cmake --build ${PWD}/timemory-build --target all --parallel 8
  cmake --build ${PWD}/timemory-build --target install --parallel 8
fi

export CMAKE_PREFIX_PATH=$PWD/timemory-install:${CMAKE_PREFIX_PATH}
export PATH=$PWD/timemory-install/bin:${PATH}
export LD_LIBRARY_PATH=$PWD/timemory-install/lib64:$PWD/timemory-install/lib:${LD_LIBRARY_PATH}
export PYTHONPATH=$PWD/timemory-install/lib/python${PYTHON_VERSION}/site-packages:${PYTHONPATH}