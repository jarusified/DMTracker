# DataMovVis: An Unified Visual Analytic Framework to study Data Movement for CUDA-enabled programs (C++/Python).

DataMovVis enables developers to study how data is exchanged between devices (e.g., CPU-GPU) for C++/Python codes that use Nvidia's CUDA programming model. 

Note: This is experimental code and is heavily under construction. So expect failures :)

## Why you might need DataMovVis framework?
A large number of code bases have started exploiting the collaborative execution models involving CPUs and GPUs. To leverage such heterogeneous executions, application developers need to allocate resources, divide the compute between CPU and GPU effectively, and ensure minimal data movement costs. Data movement across devices is a key limiting factor in heterogeneous architectures where the host (i.e., CPU) orchestrates the computation by distributing the computation workload to the devices, while the devices (i.e., CPU or GPU) execute parallel operations.  To achieve good **scalability** and **performance**, one must minimize unnecessary data movement operations and the volume of data transferred between devices.  DataMovVis enables users to visualize and analyze track data movement . 

## Setup

- Ensure you have a CUDA-capable system with [CUDA](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html) and [CUPTI](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_network) installed. 

Furthermore, the setup requires `spack` to be installed. Refer to the
documentation [spack](https://github.com/spack/spack).
```
source linux-setup.sh
```

Note: Make sure the environment variables, $FMT_SOURCE_DIR, $GOOGLETEST_SOURCE_DIR and $CUDA_SOURCE_DIR are pointing to the correct paths.

 ## Installation

```
pip install .
```

If only the C++ library needs to be installed,

```
cmake . -DFMT_SOURCE_DIR=$FMT_SOURCE_DIR -DGOOGLETEST_SOURCE_DIR=$GOOGLETEST_SOURCE_DIR -DCUDA_SOURCE_DIR=$CUDA_SOURCE_DIR
cmake --build .
```

## Usage examples (c++)

Compile the examples folder.
```
export KINETO_SOURCE_DIR=/path/to/the/dmv_profiler
cd examples
cmake . -DFMT_SOURCE_DIR=$FMT_SOURCE_DIR  -DCUDA_SOURCE_DIR=$CUDA_SOURCE_DIR -DPYBIND_SOURCE_DIR=$PYBIND_SOURCE_DIR 
make
```

This should install the binaries for the different experiments. Refer the
internal `readme` files inside each experiment for usage.


## Usage examples (python)

TODO

## Environment 

All commands have been tested on GeForce GTX 1050 TI.

* NVIDIA-SMI 515/48.07    
* Driver Version: 515.48.07    
* CUDA Version: 11.7