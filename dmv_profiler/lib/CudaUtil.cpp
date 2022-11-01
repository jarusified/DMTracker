#include "CudaUtil.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <mutex>

namespace libdmv {

bool gpuAvailable = false;

bool isGpuAvailable() {
  static std::once_flag once;
  std::call_once(once, [] {
    // determine GPU availability on the system
    cudaError_t error;
    int deviceCount;
    error = cudaGetDeviceCount(&deviceCount);
    gpuAvailable = (error == cudaSuccess && deviceCount > 0);
  });

  return gpuAvailable;
}

} // namespace libdmv
