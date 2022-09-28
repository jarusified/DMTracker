#include "pybind11/pybind11.h"
// #include "libkineto.h"
#include <iostream>

namespace py = pybind11;

PYBIND11_MODULE(dmv_profiler, m)
{
    m.doc() = "Data Movement profiler for CUDA";

    m.def("start_profile", []() { 
        // std::set<libkineto::ActivityType> types = {
        //     libkineto::ActivityType::CONCURRENT_KERNEL,
        //     libkineto::ActivityType::GPU_MEMCPY,
        //     libkineto::ActivityType::GPU_MEMSET,
        //     libkineto::ActivityType::CUDA_RUNTIME,
        //     libkineto::ActivityType::EXTERNAL_CORRELATION,
        // };

        // std::string profiler_config = "ACTIVITIES_WARMUP_PERIOD_SECS=5\n "
        //                         "CUPTI_PROFILER_METRICS=kineto__cuda_core_flops\n "
        //                         "CUPTI_PROFILER_ENABLE_PER_KERNEL=true\n "
        //                         "ACTIVITIES_DURATION_SECS=5";

        // auto &profiler = libkineto::api().activityProfiler();
        // libkineto::api().initProfilerIfRegistered();
        // profiler.prepareTrace(types, profiler_config);
        // profiler.startTrace(); 
        std::cout<<"Here";
    });
}