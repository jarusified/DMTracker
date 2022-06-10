# DataMovVis: An Unified Visual Analytic Framework to study Data Movement for CUDA-enabled programs (C++/Python).

DataMovVis enables developers to study how data is exchanged between devices (e.g., CPU-GPU) for C++/Python codes that use Nvidia's CUDA programming model. 

Note: This is experimental code and is heavily under construction. So expect failures :)

## Why you might need DataMovVis framework?
A large number of code bases have started exploiting the collaborative execution models involving CPUs and GPUs. To leverage such heterogeneous executions, application developers need to allocate resources, divide the compute between CPU and GPU effectively, and ensure minimal data movement costs. Data movement across devices is a key limiting factor in heterogeneous architectures where the host (i.e., CPU) orchestrates the computation by distributing the computation workload to the devices, while the devices (i.e., CPU or GPU) execute parallel operations.  To achieve good **scalability** and **performance**, one must minimize unnecessary data movement operations and the volume of data transferred between devices.  DataMovVis enables users to visualize and analyze track data movement . 

## Setup

- Ensure you have a CUDA-capable system and [CUPTI] installed (https://docs.nvidia.com/cuda/cupti/index.html). We can use [spack](https://github.com/spack/spack) to check this. 

```bash
spack load cuda
```


If it fails, install cuda compiler and CUPTI using [spack](https://github.com/spack/spack)
```bash
spack install cuda
```


All commands have been tested on GeForce GTX 1050 TI. Large scale experiments were run on ECP

* NVIDIA-SMI 440.33.01    
* Driver Version: 440.33.01    
* CUDA Version: 10.2

## Installation

TBD

## Usage examples

TBD
