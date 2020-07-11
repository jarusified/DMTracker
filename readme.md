# nvidia Data Movement Experiments.

## Setup

- Install [cuda toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html).
- Export the $PATH and LD_LIBRARY_PATH to the right directories.
    ```
    export LD_LIBRARY_PATH=/usr/local/cuda/lib:$LD_LIBRARY_PATH
    export PATH=$PATH:/usr/local/cuda/bin
    ```
- To validate the previous step, 
    ```
    nvcc --version
    ```
- Install cuda [nsight compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/index.html). 
- To validate if nvida nsight compute is installed.
    ```
    nv-nsight-cu-cli --version
    ```
- To run the experiments
    ```
    cd experiments/vector_addition
    make
    make profile-all
    ```
    This should dump all the results into all_metrics.txt.

All experiments have been tested on GeForce GTX 1050 TI

* NVIDIA-SMI 440.33.01    
* Driver Version: 440.33.01    
* CUDA Version: 10.2
