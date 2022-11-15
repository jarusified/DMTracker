#pragma once

#include <iostream>

#include "libdmv.h"

namespace dmv_profiler_wrapper {

class Profiler {
    
    public: 
        bool is_profiling() { return is_profiling_; }

        void init_profile();
        
        void start_profile();

        void stop_profile();

        void save_profile();

        void reset();

    private: 
        static libdmv::ActivityProfilerInterface profiler_;
        bool is_profiling_;
        std::string perf_file_path_;
};

} // namespace dmv_profiler_wrapper.