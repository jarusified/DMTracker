//////////////////////////////////////////////////////////////////////////////////////////////////////
// summary:	Gemm class
// 
// origin: SHOC (https://github.com/vetter/shocp)
////////////////////////////////////////////////////////////////////////////////////////////////////

#include "OptionParser.h"
#include "ResultDatabase.h"
// #include "Timer.h"
#include "Utility.h"
#include "cuda.h"
#include "cuda_runtime.h"
#include "cudacommon.h"
#include "cuda_fp16.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <chrono>

// Include caliper instrumentation.
#ifdef USE_CALIPER
#include <caliper/cali.h>
#include <caliper/cali_datatracker.h>
#define MARK_FUNCTION CALI_CXX_MARK_FUNCTION
#else
#define MARK_FUNCTION 
#endif

#define SEED 7

using namespace std;

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="testName">	Name of the test. </param>
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op);

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Filling memory. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">   	[in,out] If non-null,  pointer to the array to initialize. </param>
/// <param name="n">   number of elements in the array. </param>
/// <param name="maxi">	The maxi. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> void fill(T *A, int n, int maxi) {
  for (int j = 0; j < n; j++) {
      // if (std::is_same<T, double>::value)
          A[j] = rand();
          // A[j] = T((rand() % (maxi * 2 + 1)) - maxi) / T(maxi + 1.);
      // else if (std::is_same<T, half>::value)
          // A[j] = __float2half(float((rand() % (maxi * 2 + 1)) - maxi) / (maxi + 1.));
      // else
          // safe_exit(-1);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Reads a matrix. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="A">	   	[in,out] If non-null, pointer to matrix A. </param>
/// <param name="B">	   	[in,out] If non-null, pointer to matrix B. </param>
/// <param name="C">	   	[in,out] If non-null, pointer to matrix C. </param>
/// <param name="n">	   	An int to process. </param>
/// <param name="filename">	Filename of the file. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T> void readMatrix(T *A, T *B, T *C, int n, string filename) {
  std::ifstream mfs(filename.c_str());
  string line;
  // Ignore header line because it was already checked
  getline(mfs, line);
  float a, b, c;
  for (int j = 0; j < n; j++) {
    sscanf(line.c_str(), "%f %f %f", &a, &b, &c);
    A[j] = T(a);
    B[j] = T(b);
    C[j] = T(c);
  }
}

// ****************************************************************************
// Function: addBenchmarkSpecOptions
//
// Purpose:
//   Add benchmark specific options parsing.  The user is allowed to specify
//   the size of the input data in kiB.
//
// Arguments:
//   op: the options parser / parameter database
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
// Returns:  nothing
//
// ****************************************************************************
void addBenchmarkSpecOptions(OptionParser &op) {}

// ****************************************************************************
// Function: runBenchmark
//
// Purpose:
//   This benchmark measures the performance of the single precision general
//   matrix multiplication (SGEMM) operation in GFLOPS.  Data transfer time
//   over the PCIe bus is not included in this measurement.
//
// Arguments:
//  resultDB: the benchmark stores its results in this ResultDatabase
//  op: the options parser / parameter database
//
// Returns:  nothing
//
// Programmer: Anthony Danalis
// Creation: September 08, 2009
//
// Modifications:
//
// ****************************************************************************
void RunBenchmark(ResultDatabase &resultDB, OptionParser &op) {
  std::chrono::steady_clock::time_point begin = std::chrono::steady_clock::now();

  cout << "Running simple-GEMM" << endl;
  int device;
  CALI_MARK_FUNCTION_BEGIN;

  #ifdef USE_CALIPER
    CALI_MARK_BEGIN("Initialize CUDA");
  #endif

  cudaGetDevice(&device);
  cudaDeviceProp deviceProp;
  cudaGetDeviceProperties(&deviceProp, device);

  srand(SEED);

  #ifdef USE_CALIPER
    CALI_MARK_END("Initialize CUDA");
  #endif

  bool quiet = op.getOptionBool("quiet");

  if(!quiet) {
    cout << "Running single precision test" << endl;
  }
  RunTest("GEMM", resultDB, op);

  std::chrono::steady_clock::time_point end = std::chrono::steady_clock::now();
  auto totalTime = std::chrono::duration_cast<std::chrono::microseconds>(end - begin).count();
  cout<<"Total time: "<<totalTime<<endl;
  resultDB.AddResult("TotalTime", "", "microsec", totalTime);
  CALI_MARK_FUNCTION_END;
}

////////////////////////////////////////////////////////////////////////////////////////////////////
/// <summary>	Executes the test operation. </summary>
///
/// <typeparam name="T">	Generic type parameter. </typeparam>
/// <param name="testName">	Name of the test. </param>
/// <param name="resultDB">	[in,out] The result database. </param>
/// <param name="op">	   	[in,out] The operation. </param>
////////////////////////////////////////////////////////////////////////////////////////////////////

template <class T>
void RunTest(string testName, ResultDatabase &resultDB, OptionParser &op) {
  int passes = op.getOptionInt("passes");
  int device = op.getOptionInt("device");
  const bool uvm = op.getOptionBool("uvm");
  const bool uvm_prefetch = op.getOptionBool("uvm-prefetch");
  const bool uvm_advise = op.getOptionBool("uvm-advise");
  const bool uvm_prefetch_advise = op.getOptionBool("uvm-prefetch-advise");
  int kib;

  // #ifdef USE_CALIPER
  //   CALI_MARK_BEGIN("Initialize Matrix data");
  // #endif
  // Use preset problem size or read data from input file
  string filename = op.getOptionString("inputFile");
  if (filename == "") {
    int probSizes[7] = {1, 2, 4, 8, 16, 32, 64};
    kib = probSizes[op.getOptionInt("size") - 1];
  } else {
    std::ifstream mfs(filename.c_str());
    std::string line;
    char object[FIELD_LENGTH];
    sscanf(line.c_str(), "%s %d", object, &kib);
  }

  // Dimensions of matrix
  int N = kib * 1024;

  cout<<"Dimensions of matrix: "<< N << ", "<< kib<< "\n";

  #ifdef USE_CALIPER
    CALI_MARK_END("Initialize Matrix data");
  #endif

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

      if (filename == "") {
          #ifdef USE_CALIPER
            CALI_MARK_BEGIN("Fill matrix (file)");
          #endif
          fill<T>(dA, N * N, 31);
          fill<T>(dB, N * N, 31);
          fill<T>(dC, N * N, 31);
          #ifdef USE_CALIPER
            CALI_MARK_END("Fill matrix (file)");
          #endif
      } else {
          #ifdef USE_CALIPER
            CALI_MARK_BEGIN("Read matrix (file)");
          #endif
          readMatrix(dA, dB, dC, N * N, filename);
          #ifdef USE_CALIPER
            CALI_MARK_END("Read matrix (file)");
          #endif
      }
  }
  else {
      checkCudaErrors(cudaMalloc(&dA, N * N * sizeof(T)));
      checkCudaErrors(cudaMalloc(&dB, N * N * sizeof(T)));
      checkCudaErrors(cudaMalloc(&dC, N * N * sizeof(T)));

      checkCudaErrors(cudaMallocHost(&A, N * N * sizeof(T)));
      checkCudaErrors(cudaMallocHost(&B, N * N * sizeof(T)));
      checkCudaErrors(cudaMallocHost(&C, N * N * sizeof(T)));

      // Fill matrix or read from input file
      if (filename == "") {
          fill<T>(A, N * N, 31);
          fill<T>(B, N * N, 31);
          fill<T>(C, N * N, 31);
      } else {
        readMatrix(A, B, C, N * N, filename);
      }
  }
  #ifdef USE_CALIPER
    CALI_MARK_END("Initialize CUDA");
  #endif

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

  bool first = true;
  #ifdef USE_CALIPER
    CALI_CXX_MARK_LOOP_BEGIN(passesloop, "passes.loop");
  #endif
  for (int j = 0; j < passes; j++) {

      // Transfer the C matrix from GPU to CPU.
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

      string atts = "dim:" + toString(dim);
      resultDB.AddResult(testName + "-TransferTime", atts, "sec", transferTime);
      resultDB.AddResult(testName + "-KernelTime", atts, "sec", cublasTime);
      resultDB.AddResult(testName + "-TotalTime", atts, "sec", transferTime + cublasTime);
      resultDB.AddResult(testName, atts, "GFlops", cublasGflops);
      resultDB.AddResult(testName + "_PCIe", atts, "GFlops", pcieGflops);
      resultDB.AddResult(testName + "_Parity", atts, "N", transferTime / cublasTime);
      resultDB.AddOverall("GFlops", "", cublasGflops);
  }
  #ifdef USE_CALIPER
    CALI_CXX_MARK_LOOP_END(passesloop);
  #endif

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
  cublasDestroy(handle);
}


template <typename T>
__global__ void gpu_gemm_nn(int m, int n, int k,          //in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
                            T * __restrict__ dest,        //inout: pointer to C matrix data
                            const T * __restrict__ left,  //in: pointer to A matrix data
                            const T * __restrict__ right) //in: pointer to B matrix data
{
 size_t ty = blockIdx.y*blockDim.y + threadIdx.y; //global thread index Y
 size_t tx = blockIdx.x*blockDim.x + threadIdx.x; //global thread index X

 size_t n_pos = ty;
 while(n_pos < n){

  size_t m_pos = tx;
  while(m_pos < m){

   T tmp = static_cast<T>(0.0);
   for(size_t k_pos = 0; k_pos < k; ++k_pos){
    tmp += left[k_pos*m + m_pos] * right[n_pos*k + k_pos];
   }
   dest[n_pos*m + m_pos] += tmp;

   m_pos += gridDim.x*blockDim.x;
  }

  n_pos += gridDim.y*blockDim.y;
 }
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_nn(int m, int n, int k,          //in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
                               T * __restrict__ dest,        //inout: pointer to C matrix data
                               const T * __restrict__ left,  //in: pointer to A matrix data
                               const T * __restrict__ right) //in: pointer to B matrix data
{
 using int_t = int; //either int or size_t
 __shared__ T lbuf[TILE_EXT_K][TILE_EXT_M], rbuf[TILE_EXT_N][TILE_EXT_K];

 for(int_t n_pos = blockIdx.y*blockDim.y; n_pos < n; n_pos += gridDim.y*blockDim.y){ //tile offset in Y dimension

  for(int_t m_pos = blockIdx.x*blockDim.x; m_pos < m; m_pos += gridDim.x*blockDim.x){ //tile offset in X dimension

   T tmp = static_cast<T>(0.0); //accumulator

   for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
    int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

    //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
    if(m_pos + threadIdx.x < m){
     for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
      lbuf[k_loc-k_pos][threadIdx.x] = left[k_loc*m + (m_pos+threadIdx.x)];
     }
    }

    //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
    if(n_pos + threadIdx.y < n){
     for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
      rbuf[threadIdx.y][k_loc-k_pos] = right[(n_pos+threadIdx.y)*k + k_loc];
     }
    }
    __syncthreads();

    //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
    if(m_pos + threadIdx.x < m && n_pos + threadIdx.y < n){
     if(k_end - k_pos == TILE_EXT_K){ //number of loop iterations is known at compile time: Unroll it
#pragma unroll
      for(int_t l = 0; l < TILE_EXT_K; ++l){
       tmp += lbuf[l][threadIdx.x] * rbuf[threadIdx.y][l];
      }
     }else{ //number of loop iterations is not known at compile time
      for(int_t l = 0; l < (k_end - k_pos); ++l){
       tmp += lbuf[l][threadIdx.x] * rbuf[threadIdx.y][l];
      }
     }
    }
    __syncthreads();

   } //k_pos

   //Store element of the C matrix in global memory:
   if(m_pos + threadIdx.x < m && n_pos + threadIdx.y < n)
    dest[(n_pos+threadIdx.y)*m + (m_pos+threadIdx.x)] += tmp;

  } //m_pos

 } //n_pos
 return;
}


template <typename T, int TILE_EXT_N, int TILE_EXT_M, int TILE_EXT_K>
__global__ void gpu_gemm_sh_reg_nn(int m, int n, int k,          //in: matrix dimensions: C(m,n)+=A(m,k)*B(k,n)
                                   T * __restrict__ dest,        //inout: pointer to C matrix data
                                   const T * __restrict__ left,  //in: pointer to A matrix data
                                   const T * __restrict__ right) //in: pointer to B matrix data
{
 using int_t = int; //either int or size_t
 __shared__ T lbuf[TILE_EXT_K][TILE_EXT_M], rbuf[TILE_EXT_N][TILE_EXT_K];

 for(int_t n_pos = blockIdx.y*TILE_EXT_N; n_pos < n; n_pos += gridDim.y*TILE_EXT_N){ //tile offset in Y dimension
  int_t n_end = n_pos + TILE_EXT_N; if(n_end > n) n_end = n;

  for(int_t m_pos = blockIdx.x*TILE_EXT_M; m_pos < m; m_pos += gridDim.x*TILE_EXT_M){ //tile offset in X dimension
   int_t m_end = m_pos + TILE_EXT_M; if(m_end > m) m_end = m;

   if((m_end - m_pos == TILE_EXT_M) && (n_end - n_pos == TILE_EXT_N)){ //complete tile C(TILE_EXT_M,TILE_EXT_N)

    //Initialize registers to zero:
    T dreg[4][4] = {static_cast<T>(0.0)};
    T rreg[4] = {static_cast<T>(0.0)};
    T lreg[4] = {static_cast<T>(0.0)};

    for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
     int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

     //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
     for(int_t m_loc = m_pos + threadIdx.x; m_loc < m_end; m_loc += blockDim.x){
      for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
       lbuf[k_loc - k_pos][m_loc - m_pos] = left[k_loc*m + m_loc];
      }
     }

     //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
     for(int_t n_loc = n_pos + threadIdx.y; n_loc < n_end; n_loc += blockDim.y){
      for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
       rbuf[n_loc - n_pos][k_loc - k_pos] = right[n_loc*k + k_loc];
      }
     }
     __syncthreads();

     //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
     if(k_end - k_pos == TILE_EXT_K){
#pragma unroll
      for(int_t l = 0; l < TILE_EXT_K; ++l){
#pragma unroll
       for(int_t j = 0; j < 4; ++j) rreg[j] = rbuf[threadIdx.y + blockDim.y*j][l];
#pragma unroll
       for(int_t j = 0; j < 4; ++j) lreg[j] = lbuf[l][threadIdx.x + blockDim.x*j];
#pragma unroll
       for(int_t j = 0; j < 4; ++j){
#pragma unroll
        for(int_t i = 0; i < 4; ++i){
         dreg[j][i] += lreg[i] * rreg[j];
        }
       }
      }
     }else{
      for(int_t l = 0; l < (k_end - k_pos); ++l){
#pragma unroll
       for(int_t j = 0; j < 4; ++j) rreg[j] = rbuf[threadIdx.y + blockDim.y*j][l];
#pragma unroll
       for(int_t j = 0; j < 4; ++j) lreg[j] = lbuf[l][threadIdx.x + blockDim.x*j];
#pragma unroll
       for(int_t j = 0; j < 4; ++j){
#pragma unroll
        for(int_t i = 0; i < 4; ++i){
         dreg[j][i] += lreg[i] * rreg[j];
        }
       }
      }
     }
     __syncthreads();

    } //k_pos

    //Store elements of the C matrix in global memory:
#pragma unroll
    for(int_t j = 0; j < 4; ++j){
#pragma unroll
     for(int_t i = 0; i < 4; ++i){
      dest[(n_pos + threadIdx.y + blockDim.y*j)*m + (m_pos + threadIdx.x + blockDim.x*i)] += dreg[j][i];
     }
    }

   }else{ //incomplete tile of C

    //Initialize registers to zero:
    T dreg[4][4] = {static_cast<T>(0.0)};
    T rreg[4] = {static_cast<T>(0.0)};
    T lreg[4] = {static_cast<T>(0.0)};

    for(int_t k_pos = 0; k_pos < k; k_pos += TILE_EXT_K){ //k_pos is the position of the CUDA thread along the K dimension
     int_t k_end = k_pos + TILE_EXT_K; if(k_end > k) k_end = k;

     //Load a tile of matrix A(m_pos:TILE_EXT_M, k_pos:TILE_EXT_K):
     for(int_t m_loc = m_pos + threadIdx.x; m_loc < m_end; m_loc += blockDim.x){
      for(int_t k_loc = k_pos + threadIdx.y; k_loc < k_end; k_loc += blockDim.y){
       lbuf[k_loc - k_pos][m_loc - m_pos] = left[k_loc*m + m_loc];
      }
     }

     //Load a tile of matrix B(k_pos:TILE_EXT_K, n_pos:TILE_EXT_N):
     for(int_t n_loc = n_pos + threadIdx.y; n_loc < n_end; n_loc += blockDim.y){
      for(int_t k_loc = k_pos + threadIdx.x; k_loc < k_end; k_loc += blockDim.x){
       rbuf[n_loc - n_pos][k_loc - k_pos] = right[n_loc*k + k_loc];
      }
     }
     __syncthreads();

     //Multiply two loaded tiles to produce a tile of matrix C(m_pos:TILE_EXT_M,n_pos:TILE_EXT_N):
     for(int_t l = 0; l < (k_end - k_pos); ++l){
      for(int_t i = 0, j = threadIdx.y; j < n_end - n_pos; j += blockDim.y, i++) rreg[i] = rbuf[j][l];
      for(int_t i = 0, j = threadIdx.x; j < m_end - m_pos; j += blockDim.x, i++) lreg[i] = lbuf[l][j];
#pragma unroll
      for(int_t j = 0; j < 4; ++j){
#pragma unroll
       for(int_t i = 0; i < 4; ++i){
        dreg[j][i] += lreg[i] * rreg[j];
       }
      }
     }
     __syncthreads();

    } //k_pos

    //Store element of the C matrix in global memory:
    for(int_t j = 0, n_loc = n_pos + threadIdx.y; n_loc < n_end; n_loc += blockDim.y, j++){
     for(int_t i = 0, m_loc = m_pos + threadIdx.x; m_loc < m_end; m_loc += blockDim.x, i++){
      dest[n_loc*m + m_loc] += dreg[j][i];
     }
    }

   }

  } //m_pos

 } //n_pos
 return;
}