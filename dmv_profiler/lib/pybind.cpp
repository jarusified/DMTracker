#include "pybind11/pybind11.h"

#include "libdmv.h"

PYBIND11_MODULE(dmv_profiler, m) {
  m.doc() = "Data Movement profiler for CUDA";

  m.def("start_profile", []() {
    std::set<libdmv::ActivityType> types = {
        libdmv::ActivityType::CONCURRENT_KERNEL,
        libdmv::ActivityType::GPU_MEMCPY,
        libdmv::ActivityType::GPU_MEMSET,
        libdmv::ActivityType::CUDA_RUNTIME,
        libdmv::ActivityType::EXTERNAL_CORRELATION,
    };

    std::string profiler_config =
        "ACTIVITIES_WARMUP_PERIOD_SECS=5\n "
        "CUPTI_PROFILER_METRICS=kineto__cuda_core_flops\n "
        "CUPTI_PROFILER_ENABLE_PER_KERNEL=true\n "
        "ACTIVITIES_DURATION_SECS=5";

    auto &profiler = libdmv::api().activityProfiler();
    libdmv::api().initProfilerIfRegistered();
    profiler.prepareTrace(types, profiler_config);
    profiler.startTrace();
  });
}