#include <memory>
#include <mutex>

#include "ActivityProfilerProxy.h"
#include "Config.h"

#ifdef HAS_CUPTI
#include "CuptiActivityApi.h"
#include "CuptiCallbackApi.h"
#include "CuptiNvmlGpuUtilization.h"
#include "CuptiRangeProfiler.h"
#include "EventProfilerController.h"
#include "cupti_call.h"
#endif
#include "libdmv.h"

#include "Logger.h"

namespace libdmv {

#ifdef HAS_CUPTI
static bool initialized = false;
static std::mutex initMutex;
static void initProfilers(CUpti_CallbackDomain /*domain*/,
                          CUpti_CallbackId /*cbid*/,
                          const CUpti_CallbackData *cbInfo) {
  CUpti_ResourceData *d = (CUpti_ResourceData *)cbInfo;
  CUcontext ctx = d->context;

  VLOG(0) << "CUDA Context created";
  std::lock_guard<std::mutex> lock(initMutex);

  if (!initialized) {
    libdmv::api().initProfilerIfRegistered();
    initialized = true;
    LOG(INFO) << "libdmv profilers activated";
  }
  if (getenv("DMV_DISABLE_EVENT_PROFILER") != nullptr) {
    VLOG(0) << "Event profiler disabled via env var";
  } else {
    ConfigLoader &config_loader = libdmv::api().configLoader();
    config_loader.initBaseConfig();
    EventProfilerController::start(ctx, config_loader);
  }
}

// Some models suffer from excessive instrumentation code gen
// on dynamic attach which can hang for more than 5+ seconds.
// If the workload was meant to be traced, preload the CUPTI
// to take the performance hit early on.
// https://docs.nvidia.com/cupti/r_main.html#r_overhead
static bool shouldPreloadCuptiInstrumentation() {
#if defined(CUDA_VERSION) && CUDA_VERSION < 11020
  return true;
#else
  return false;
#endif
}

static void stopProfiler(CUpti_CallbackDomain /*domain*/,
                         CUpti_CallbackId /*cbid*/,
                         const CUpti_CallbackData *cbInfo) {
  CUpti_ResourceData *d = (CUpti_ResourceData *)cbInfo;
  CUcontext ctx = d->context;

  LOG(INFO) << "CUDA Context destroyed";
  std::lock_guard<std::mutex> lock(initMutex);
  EventProfilerController::stop(ctx);
}

static std::unique_ptr<CuptiRangeProfilerInit> rangeProfilerInit;
static std::unique_ptr<CuptiNvmlGpuUtilization> gpuUtilizationInit;
#endif // HAS_CUPTI

} // namespace libdmv

// Callback interface with CUPTI and library constructors
using namespace libdmv;
extern "C" {

// Return true if no CUPTI errors occurred during init
void libdmv_init(bool cpuOnly, bool logOnError) {
#ifdef HAS_CUPTI
  LOG(INFO) << "CUPTI instrumentation enabled.";
  if (!cpuOnly) {
    // libcupti will be lazily loaded on this call.
    // If it is not available (e.g. CUDA is not installed),
    // then this call will return an error and we just abort init.
    auto &cbapi = CuptiCallbackApi::singleton();
    bool status = false;
    bool initRangeProfiler = true;
    bool initGpuUtilization = true;

    if (cbapi.initSuccess()) {
      const CUpti_CallbackDomain domain = CUPTI_CB_DOMAIN_RESOURCE;
      status = cbapi.registerCallback(
          domain, CuptiCallbackApi::CUDA_LAUNCH_KERNEL, initProfilers);
      status =
          status && cbapi.registerCallback(
                        domain, CuptiCallbackApi::RESOURCE_CONTEXT_DESTROYED,
                        stopProfiler);

      if (status) {
        status = cbapi.enableCallback(
            domain, CuptiCallbackApi::RESOURCE_CONTEXT_CREATED);
        status =
            status && cbapi.enableCallback(
                          domain, CuptiCallbackApi::RESOURCE_CONTEXT_DESTROYED);
      }
    }

    if (!cbapi.initSuccess() || !status) {
      initRangeProfiler = false;
      cpuOnly = true;
      if (logOnError) {
        CUPTI_CALL(cbapi.getCuptiStatus());
        LOG(WARNING) << "CUPTI initialization failed - "
                     << "CUDA profiler activities will be missing";
        LOG(INFO)
            << "If you see CUPTI_ERROR_INSUFFICIENT_PRIVILEGES, refer to "
            << "https://developer.nvidia.com/"
               "nvidia-development-tools-solutions-err-nvgpuctrperm-cupti";
      }
    }

    // initialize CUPTI Range Profiler API
    if (initRangeProfiler) {
      rangeProfilerInit = std::make_unique<CuptiRangeProfilerInit>();
    }

    // if (initGpuUtilization) {
      // int dev {};
      // cudaGetDevice(&dev);
      // cudaSetDevice(dev);

      // std::string const filename = { "data/gpuStats.csv" };

      // Create NVML class to retrieve GPU stats
      // CuptiNvmlGpuUtilization nvml(0, filename);

      // Create thread to gather GPU stats
      // std::thread threadStart(&CuptiNvmlGpuUtilization::getStats_temp, &nvml);
    // }
  }

  if (shouldPreloadCuptiInstrumentation()) {
    CuptiActivityApi::forceLoadCupti();
  }

#endif // HAS_CUPTI

  ConfigLoader &config_loader = libdmv::api().configLoader();
  libdmv::api().registerProfiler(
      std::make_unique<ActivityProfilerProxy>(cpuOnly, config_loader));
}

// void libdmv_fin(bool cpuOnly, bool logOnError) {
//   if (!cpuOnly) {
//     std::thread threadKill(&CuptiNvmlGpuUtilization::killThread, &nvml);
//     threadStart.join();
//     threadKill.join();
//   }
// }

void suppresslibdmvLogMessages() { SET_LOG_SEVERITY_LEVEL(ERROR); }

} // extern C
