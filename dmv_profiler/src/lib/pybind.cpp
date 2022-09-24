#include "pybind11/pybind11.h"

#include "libkineto.h"

PYBIND11_MODULE(dmv_profiler, m)
{
    m.doc() = "Data Movement profiler for CUDA";

    m.def(
        "add_event",
        [](std::string eventname, py::dict metadata)
        {
            nl::json json_metadata = metadata;
            torch_rdu::JIT::jit().profiler().add_event(eventname, json_metadata);
        },
        py::arg("eventname") = py::none(), py::arg("metadata") = py::dict());
    m.def("end_profile", []()
          { torch_rdu::JIT::jit().profiler().end_profile(); });
    m.def("save_profile", [](std::string filename)
          { torch_rdu::JIT::jit().profiler().save_profile(filename); });
    m.def("start_profile", []() { 
        std::set<libkineto::ActivityType> types = {
            libkineto::ActivityType::CONCURRENT_KERNEL,
            libkineto::ActivityType::GPU_MEMCPY,
            libkineto::ActivityType::GPU_MEMSET,
            libkineto::ActivityType::CUDA_RUNTIME,
            libkineto::ActivityType::EXTERNAL_CORRELATION,
        };

        std::string profiler_config = "ACTIVITIES_WARMUP_PERIOD_SECS=5\n "
                                "CUPTI_PROFILER_METRICS=kineto__cuda_core_flops\n "
                                "CUPTI_PROFILER_ENABLE_PER_KERNEL=true\n "
                                "ACTIVITIES_DURATION_SECS=5";

        auto &profiler = libkineto::api().activityProfiler();
        libkineto::api().initProfilerIfRegistered();
        profiler.prepareTrace(types, profiler_config);
        profiler.startTrace(); 
    });
    m.def(
        "start_event",
        [](std::string eventname, py::dict metadata)
        {
            nl::json json_metadata = metadata;
            torch_rdu::JIT::jit().profiler().start_event(eventname, json_metadata);
        },
        py::arg("eventname") = py::none(), py::arg("metadata") = py::dict());
    m.def("end_event", [](std::string eventname)
          { torch_rdu::JIT::jit().profiler().end_event(eventname); });
}