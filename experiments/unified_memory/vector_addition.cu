#include <stdio.h>
#include <stdlib.h>
#include <iostream>

#ifdef __DRIVER_TYPES_H__
static const char *_cudaGetErrorEnum(cudaError_t error) {
  return cudaGetErrorName(error);
}
#endif

// Size of array = 1M or 2**20
#define N 1048576

// Kernel function to add the elements of two arrays
__global__ void add_vectors(int n, float *a, float *b)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < n) b[id] = a[id] + b[id];
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

template <typename T>
void check(T result, char const *const func, const char *const file, int const line) {
  if (result) {
    fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, 
    static_cast<unsigned int>(result), _cudaGetErrorEnum(result), func);
    exit(EXIT_FAILURE);
  }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)
 
int main(void)
{
  // Number of bytes to allocate for N integers
  size_t bytes = N*sizeof(float);
  
  // Allocate memory for arrays A, B, and C on host
  float *A, *B;
 
  // Allocate Unified Memory – accessible from CPU or GPU
  checkCudaErrors(cudaMallocManaged(&A, bytes));
  checkCudaErrors(cudaMallocManaged(&B, bytes));
 
  // initialize x and y arrays on the host
  for (int i = 0; i < N; i++) {
    A[i] = 1.0f;
    B[i] = 2.0f;
  }
 
  // Prefetch the data to the GPU
  char *prefetch = getenv("__PREFETCH");
  if (prefetch == NULL || strcmp(prefetch, "off") != 0) {
    int device = -1;
    checkCudaErrors(cudaGetDevice(&device));
    checkCudaErrors(cudaMemPrefetchAsync(A, bytes, device, NULL));
    checkCudaErrors(cudaMemPrefetchAsync(B, bytes, device, NULL));
  }
 
  // Run kernel on 1M elements on the GPU
  //		blockSize: number of CUDA threads per grid block
	//		numBlocks: number of blocks in grid
  int blockSize = 256;
  int numBlocks = (N + blockSize - 1) / blockSize;
  // int numBlocks = ceil( float(N) / blockSize);

  // add_vectors_w_stride_no_c<<<numBlocks, blockSize>>>(N, A, B);
  // add_vectors_w_stride<<<numBlocks, blockSize>>>(N, A, B, C);
  add_vectors<<<numBlocks, blockSize>>>(N, A, B);

  // Wait for GPU to finish before accessing on host
  checkCudaErrors(cudaDeviceSynchronize());
 
  // Check for errors (all values should be 3.0f)
  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    maxError = fmax(maxError, fabs(B[i]-3.0f));
  std::cout << "Max error: " << maxError << std::endl;
 
  // Free memory
  checkCudaErrors(cudaFree(A));
  checkCudaErrors(cudaFree(B));

  printf("\n---------------------------\n");
	printf("__SUCCESS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", blockSize);
	printf("Blocks In Grid    = %d\n", numBlocks);
  printf("---------------------------\n\n");
  
  return 0;
}