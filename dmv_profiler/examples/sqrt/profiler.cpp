#include <iostream>
#include <set>
#include <string>

#include "profiler.h"

namespace dmv_profiler_wrapper {

    void Profiler::init_profile() {
        std::set<libdmv::ActivityType> types = {
            libdmv::ActivityType::CONCURRENT_KERNEL,
            libdmv::ActivityType::GPU_MEMCPY,
            libdmv::ActivityType::GPU_MEMSET,
            libdmv::ActivityType::CUDA_RUNTIME,
            libdmv::ActivityType::EXTERNAL_CORRELATION,
            libdmv::ActivityType::CPU_OP,
            libdmv::ActivityType::DEVICE,
            libdmv::ActivityType::DRIVER
        };

        std::string profiler_config = "EVENTS=active_cycles\n "
                                    "METRICS=ipc\n "
                                    "CUPTI_PROFILER_METRICS=l1tex__data_bank_conflicts_pipe_lsu,sm__inst_executed\n "
                                    "CUPTI_PROFILER_ENABLE_PER_KERNEL=true\n ";

        profiler_ = libdmv::api().activityProfiler();
        libdmv::api().initProfilerIfRegistered();
        profiler_.prepareTrace(types, profiler_config);

        is_profiling_ = true;        
    }

    void Profiler::start_profile() {
        profiler_.startTrace();
    }

    void Profiler::stop_profile() {
        auto trace = profiler_.stopTrace();
        std::cout << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.\n";
        std::string filePath = "./" + perf_file_path_;
        trace->save(filePath);
    }

}
