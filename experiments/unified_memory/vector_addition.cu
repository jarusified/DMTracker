#include<stdio.h>
#include<stdlib.h>
#include <iostream>

// Size of array = 1M or 2**20
#define N 1048576

// Kernel function to add the elements of two arrays
__global__ void add_vectors(float *a, float *b, float *c)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
}

// Kernel function to add the elements of two arrays with stride.
__global__ void add_vectors_w_stride(int n, float *a, float *b, float *c)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    c[i] = a[i] + b[i];
}

// Kernel function to add the elements of two arrays with stride.
__global__ void add_vectors_w_stride_no_c(int n, float *a, float *b)
{
  int index = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = index; i < n; i += stride)
    b[i] = a[i] + b[i];
}

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}
 
int main(void)
{
  // Number of bytes to allocate for N integers
  size_t bytes = N*sizeof(float);
  
  // Allocate memory for arrays A, B, and C on host
  float *A, *B;
 
  // Allocate Unified Memory – accessible from CPU or GPU
  gpuErrchk (cudaMallocManaged(&A, bytes));
  gpuErrchk (cudaMallocManaged(&B, bytes));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }
 
  // // Prefetch the data to the GPU
  // char *prefetch = getenv("__PREFETCH");
  // if (prefetch == NULL || strcmp(prefetch, "off") != 0) {
  //   int device = -1;
  //   cudaGetDevice(&device);
  //   cudaMemPrefetchAsync(A, bytes, device, NULL);
  //   cudaMemPrefetchAsync(B, bytes, device, NULL);
  // }
 
  // Run kernel on 1M elements on the GPU
  //		blockSize: number of CUDA threads per grid block
	//		numBlocks: number of blocks in grid
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  // int numBlocks = ceil( float(N) / blockSize);
  add_vectors_w_stride_no_c<<<numBlocks, blockSize>>>(N, A, B);

  // add_vectors_w_stridev<<<numBlocks, blockSize>>>(N, A, B, C);
  
  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(B[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  cudaFree(A);
  cudaFree(B);
  // cudaFree(C);

  printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", blockSize);
	printf("Blocks In Grid    = %d\n", numBlocks);
  printf("---------------------------\n\n");
  
  return 0;
}