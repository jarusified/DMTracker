# How to run this experiment.

1. Clone the cutlass github repo.

```
git clone https://github.com/NVIDIA/cutlass.git
```

2. Follow instructions to install the library. See [link](https://github.com/NVIDIA/cutlass/blob/master/README.md)

3. Export the CUTLASS_SOURCE_DIR to point to the location of the repo.

```
export CUTLASS_SOURCE_DIR=$(pwd)
```

4. Build the gemm example with the library.
```
cmake -
```