#include <stdio.h>
#include <iostream>
#include <cuda.h>
#include <cupti.h>
#include <cuda_runtime.h>

#define DRIVER_API_CALL(apiFuncCall)                                           \
do {                                                                           \
    CUresult _status = apiFuncCall;                                            \
    if (_status != CUDA_SUCCESS) {                                             \
        fprintf(stderr, "%s:%d: error: function %s failed with error %d.\n",   \
                __FILE__, __LINE__, #apiFuncCall, _status);                    \
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define RUNTIME_API_CALL(apiFuncCall)                                          \
do {                                                                           \
    cudaError_t _status = apiFuncCall;                                         \
    if (_status != cudaSuccess) {                                              \
        fprintf(stderr, "%s:%d: error: function %s failed with error %s.\n",   \
                __FILE__, __LINE__, #apiFuncCall, cudaGetErrorString(_status));\
        exit(-1);                                                              \
    }                                                                          \
} while (0)

#define USE_CUPTI 1

extern void initTrace(void);
extern void finiTrace(void);

// Size of array = 1M or 2**20
#define N 1048576*256

// Kernel function to add the elements of two arrays
__global__ void add_vectors(float *a, float *b, float *c)
{
	int id = blockDim.x * blockIdx.x + threadIdx.x;
	if(id < N) c[id] = a[id] + b[id];
}

// Main program
int main(void) {
	// initialize the activity trace
	// make sure activity is enabled before any CUDA API
	#if USE_CUPTI
		initTrace();
		DRIVER_API_CALL(cuInit(0));
	#endif

	// Number of bytes to allocate for N integers
	size_t bytes = N*sizeof(float);

	// Allocate memory for arrays A, B, and C on host
	float *A = (float*)malloc(bytes);
	float *B = (float*)malloc(bytes);
	float *C = (float*)malloc(bytes);

	// Allocate memory for arrays d_A, d_B, and d_C on device
	float *d_A, *d_B, *d_C;
	RUNTIME_API_CALL(cudaMalloc(&d_A, bytes));
	RUNTIME_API_CALL(cudaMalloc(&d_B, bytes));
	RUNTIME_API_CALL(cudaMalloc(&d_C, bytes));

	// Fill host arrays A and B
	for(int i = 0; i < N; i++){
		A[i] = 1.0f;
		B[i] = 2.0f;
	}

	// Copy data from host arrays A and B to device arrays d_A and d_B
	RUNTIME_API_CALL(cudaMemcpy(d_A, A, bytes, cudaMemcpyHostToDevice));
	RUNTIME_API_CALL(cudaMemcpy(d_B, B, bytes, cudaMemcpyHostToDevice));

	// Set execution configuration parameters
	//		blockSize: number of CUDA threads per grid block
	//		numBlocks: number of blocks in grid
	int blockSize = 256;
	int numBlocks = ceil( float(N) / blockSize );

	// Launch kernel
	add_vectors<<< numBlocks, blockSize >>>(d_A, d_B, d_C);

	// Copy data from device array d_C to host array C
	RUNTIME_API_CALL(cudaMemcpy(C, d_C, bytes, cudaMemcpyDeviceToHost));

	// Check for errors (all values should be 3.0f)
	float maxError = 0.0f;
	for (int i = 0; i < N; i++) {
		maxError = fmax(maxError, fabs(C[i] - 3.0f));

		if(C[i] != 3.0f){ 
			std::cout << A[i] << B[i] << C[i] << std::endl;
			printf("\nError: value of C[%d] = %f instead of 3\n\n", i, C[i]);
			exit(-1);
		}
	}
	
	// Free CPU memory
	free(A);
	free(B);
	free(C);

	// Free GPU memory
	RUNTIME_API_CALL(cudaFree(d_A));
	RUNTIME_API_CALL(cudaFree(d_B));
	RUNTIME_API_CALL(cudaFree(d_C));

  	printf("\n---------------------------\n");
	printf("__RESULTS__\n");
	printf("---------------------------\n");
	printf("N                 = %d\n", N);
	printf("Threads Per Block = %d\n", blockSize);
	printf("Blocks In Grid    = %d\n", numBlocks);
	printf("Error             = %f\n", maxError);
  	printf("---------------------------\n\n");

	#if USE_CUPTI
	  	// Flush all remaining CUPTI buffers before resetting the device.
	  	// This can also be called in the cudaDeviceReset callback.
	  	cuptiActivityFlushAll(0);
  	#endif

	return 0;
}
