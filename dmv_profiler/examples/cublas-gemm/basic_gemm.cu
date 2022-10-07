#include "OptionParser.h"
#include "ResultDatabase.h"
#include "Utility.h"
#include "cudacommon.h"
#include "kernel_header.h"

#include "cuda.h"
#include "cuda_runtime.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

#define SEED 7

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

template <class T>
inline void basicGEMM(int version,
					T alpha,
					const T *A,
					const T *B,
					T beta,
					T *C, int N);

template <class T>
void fill(T *A, int m, int n) {
  for (int j = 0; j < m * n; j++) {
	  A[j] = rand();
  }
}

#define IDX2F(i,j,ld) ((((j)-1)*(ld))+((i)-1))
#define IDX2C(i,j,ld) (((j)*(ld))+(i))

template <class T>
void fill_1_index(T *A, int m, int n) {
  for (int j = 1; j <= n; j++) {
	for (int i = 1; i <= m; i++) {
	  A[IDX2F(i, j, m)] = (float)((i - 1) * m + j);
	}
  }
}

template <class T>
void fill_0_index(T *A, int m, int n) {
  for (int j = 0; j < n; j++) {
	for (int i = 0; i < m; i++) {
	  A[IDX2C(i, j, m)] = (float)(i * m + j + 1);
	}
  }
}

void addBenchmarkSpecOptions(OptionParser &op) {}

void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  cout << "Running GEMM" << endl;
  int device;

  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  srand(SEED);

  bool quiet = op.getOptionBool("quiet");

  if(!quiet) {
	cout << "Running single precision test" << endl;
  }
  RunTest<float>("basicGEMM", resultDB, op);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  resultDB.AddResult("TotalTime", "", "microsec", std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count());
}

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op) {
  int passes = op.getOptionInt("passes");
  int device = op.getOptionInt("device");
  int field_length = op.getOptionInt("field-length");
  int kernel_version = op.getOptionInt("kernel-version");
  string fill_strategy = op.getOptionString("fill-strategy");
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
  int kib;

  std::cout<<"Field length: " <<field_length<<std::endl;

  string filename = op.getOptionString("inputFile");
  if (filename == "") {
	int probSizes[10] = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512};
	kib = probSizes[op.getOptionInt("size") - 1];
  } else {
	std::ifstream mfs(filename.c_str());
	std::string line;
	char object[field_length];
	sscanf(line.c_str(), "%s %d", object, &kib);
  }

  // Dimensions of matrix
  int N = kib * 1024;

  cout<<"Dimensions of matrix: "<< N << ", "<< kib<< "\n";

  // Allocate GPU memory
  T *dA, *dB, *dC;
  T *A;
  T *B;
  T *C;
  if (uvm || uvm_prefetch || uvm_advise || uvm_prefetch_advise) {
	  checkCudaErrors(cudaMallocManaged(&dA, N * N* sizeof(T)));
	  checkCudaErrors(cudaMallocManaged(&dB, N * N* sizeof(T)));
	  checkCudaErrors(cudaMallocManaged(&dC, N * N* sizeof(T)));

	  #ifdef USE_CALIPER
		CALI_DATATRACKER_TRACK(dA, sizeof(T)* N * N);
		CALI_DATATRACKER_TRACK(dA, sizeof(T)* N * N);
	  #endif
  }
  else {
	  checkCudaErrors(cudaMalloc(&dA, N * N * sizeof(T)));
	  checkCudaErrors(cudaMalloc(&dB, N * N * sizeof(T)));
	  checkCudaErrors(cudaMalloc(&dC, N * N * sizeof(T)));

	  checkCudaErrors(cudaMallocHost(&A, N * N * sizeof(T)));
	  checkCudaErrors(cudaMallocHost(&B, N * N * sizeof(T)));
	  checkCudaErrors(cudaMallocHost(&C, N * N * sizeof(T)));
  }
  // Fill matrix or read from input file
  if (fill_strategy == "column-format") {
    fill_1_index<T>(A, N, N);
    fill_1_index<T>(B, N, N);
    fill_1_index<T>(C, N, N);
  } else if (fill_strategy == "default") {
    fill<T>(A, N, N);
    fill<T>(B, N, N);
    fill<T>(C, N, N);
  }
 

  // Copy input to GPU
  cudaEvent_t start, stop;
  checkCudaErrors(cudaEventCreate(&start));
  checkCudaErrors(cudaEventCreate(&stop));
  float elapsedTime;

  // Copy inputs to GPU

  double transferTime = 0;
  checkCudaErrors(cudaEventRecord(start, 0));

  if (uvm) {
	  // Do nothing
  } else if (uvm_prefetch) {
	  // could ignore this to test demand paging performance affect
	  checkCudaErrors(cudaMemPrefetchAsync(dA, N * N * sizeof(T), device));
	  cudaStream_t s1;
	  checkCudaErrors(cudaStreamCreate(&s1));
	  checkCudaErrors(cudaMemPrefetchAsync(dB, N * N * sizeof(T), device, s1));
	  checkCudaErrors(cudaStreamDestroy(s1));
	  // checkCudaErrors(cudaStreamSynchronize(0));
	  // checkCudaErrors(cudaStreamSynchronize((cudaStream_t)1));
  } else if (uvm_advise) {
	  // Do nothing for demand paging
	  checkCudaErrors(cudaMemAdvise(dA, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
	  checkCudaErrors(cudaMemAdvise(dB, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
  } else if (uvm_prefetch_advise) {
	  checkCudaErrors(cudaMemAdvise(dA, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
	  checkCudaErrors(cudaMemAdvise(dB, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, device));
	  checkCudaErrors(cudaMemPrefetchAsync(dA, N * N * sizeof(T), device));
	  cudaStream_t s1;
	  checkCudaErrors(cudaStreamCreate(&s1));
	  checkCudaErrors(cudaMemPrefetchAsync(dB, N * N * sizeof(T), device, s1));
	  checkCudaErrors(cudaStreamDestroy(s1));
  } else {
	  checkCudaErrors(cudaMemcpy(dA, A, N * N * sizeof(T), cudaMemcpyHostToDevice));
	  checkCudaErrors(cudaMemcpy(dB, B, N * N * sizeof(T), cudaMemcpyHostToDevice));
  }

  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  cudaEventElapsedTime(&elapsedTime, start, stop);
  transferTime += elapsedTime * 1.e-3;

  float kernelTime = 0.0f;
  float alpha = 1.0f, beta = 0.0f;

  checkCudaErrors(cudaEventRecord(start, 0));
  basicGEMM<T>(kernel_version, alpha, dA, dB, beta, dC, N);
  checkCudaErrors(cudaEventRecord(stop, 0));
  checkCudaErrors(cudaEventSynchronize(stop));
  float currTime = 0.0f;
  checkCudaErrors(cudaEventElapsedTime(&currTime, start, stop));
  kernelTime += currTime;

	  checkCudaErrors(cudaEventRecord(start, 0));    // timing may be affected by async

	  if (uvm) {
		// Do nothing
	  } else if (uvm_prefetch) {
		  checkCudaErrors(cudaMemPrefetchAsync(dC, N * N * sizeof(T), cudaCpuDeviceId));
		  // checkCudaErrors(cudaStreamSynchronize(0));
	  } else if (uvm_advise) {
		  checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
		  checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
	  } else if (uvm_prefetch_advise) {
		  checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
		  checkCudaErrors(cudaMemAdvise(dC, N * N * sizeof(T), cudaMemAdviseSetPreferredLocation, cudaCpuDeviceId));
		  checkCudaErrors(cudaMemPrefetchAsync(dC, N * N * sizeof(T), cudaCpuDeviceId));
	  } else {
		  checkCudaErrors(cudaMemcpy(C, dC, N * N * sizeof(T), cudaMemcpyDeviceToHost));
	  }

	  checkCudaErrors(cudaEventRecord(stop, 0));
	  checkCudaErrors(cudaEventSynchronize(stop));
	  float oTransferTime = 0.0f;
	  checkCudaErrors(cudaEventElapsedTime(&oTransferTime, start, stop));
	  oTransferTime *= 1.e-3;

	  // Add the PCIe transfer time to total transfer time only once
	  transferTime += oTransferTime;

	//   double cublasGflops = 2. * m * n * k / cublasTime / 1e9;
	//   double pcieGflops = 2. * m * n * k / (cublasTime + transferTime) / 1e9;
	  std::string transb_string = "";
	  string atts = "dim:" + toString(N);
	  resultDB.AddResult(testName + "-" + transb_string + "-TransferTime", atts, "sec", transferTime);
	//   resultDB.AddResult(testName + "-" + transb_string + "-KernelTime", atts, "sec", cublasTime);
	  resultDB.AddResult(testName + "-" + transb_string + "-TotalTime", atts, "sec", transferTime);
	//   resultDB.AddResult(testName + "-" + transb_string, atts, "GFlops", cublasGflops);
	//   resultDB.AddResult(testName + "-" + transb_string + "_PCIe", atts, "GFlops", pcieGflops);
	  resultDB.AddResult(testName + "-" + transb_string + "_Parity", atts, "N", transferTime);
	//   resultDB.AddOverall("GFlops", "", cublasGflops);

  checkCudaErrors(cudaFree(dA));
  checkCudaErrors(cudaFree(dB));
  checkCudaErrors(cudaFree(dC));

  if (!uvm && !uvm_prefetch && !uvm_advise && !uvm_prefetch_advise) {
	checkCudaErrors(cudaFreeHost(A));
	checkCudaErrors(cudaFreeHost(B));
	checkCudaErrors(cudaFreeHost(C));
  }

  checkCudaErrors(cudaEventDestroy(start));
  checkCudaErrors(cudaEventDestroy(stop));
}


template <>
inline void basicGEMM<float>(int version,
							float alpha,
							const float *dA,
							const float *dB,
							float beta,
							float *dC,
							int N) {

	std::cout<<"Kernel version: "<<version<<std::endl;
	if(version == 0)
	{
		dim3 blocks(N/32, N/32);
		dim3 threads(32, 32);
		sgemm_v0<<<blocks, threads>>>(alpha, dA, dB, beta, dC, N);
	}
	else if(version == 1)
	{
		dim3 blocks(N/32, N/32);
		dim3 threads(32, 32);
		sgemm_v1<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC);
	}
	else if(version == 2)
	{
		dim3 blocks(N/32, N/32);
	 	dim3 threads(32, 32);
		sgemm_v2<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC);
	}
	else if(version == 3)
	{
		dim3 blocks(N/32, N/32);
		dim3 threads(8, 32);
	 	sgemm_v3<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC);
	}
	else if(version == 4)
	{
		dim3 blocks(N/32, N/32);
		dim3 threads(8, 32);
		sgemm_v4<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC);
	}
	else if(version == 5)
	{
		dim3 blocks(N/128, N/128);
	 	int threads = 256;
	 	sgemm_v5<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC);
	}
	// else if(version == 6)
	// {
	// 	dim3 blocks(N/128, N/128);
	// 	int threads = 256;
	// 	sgemm_v6<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC, N);
	// }
	// else if(version == 7)
	// {
	// 	dim3 blocks(N/128, N/128);
	// 	int threads = 256;
	// 	sgemm_v7<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC, N);
	// }
	// else if(version == 8)
	// {
	// 	dim3 blocks(N/128, N/128);
	// 	int threads = 256;
	// 	sgemm_v8<<<blocks, threads>>>(N, N, N, alpha, dA, dB, beta, dC, N);
	// }

}
