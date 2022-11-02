#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

#include <nvml.h>

int constexpr size_of_vector { 100000 };
int constexpr nvml_device_name_buffer_size { 100 };

namespace libdmv {
class CuptiNvmlGpuUtilization {
  public:
    CuptiNvmlGpuUtilization(int const &deviceID, std::string const &filename);
    ~CuptiNvmlGpuUtilization();
    void getStats();
    void killThread();

  private:
    typedef struct _stats {
        std::time_t        timestamp;
        uint               temperature;
        uint               powerUsage;
        uint               powerLimit;
        nvmlUtilization_t  utilization;
        nvmlMemory_t       memory;
        unsigned long long throttleReasons;
        uint               clockSM;
        uint               clockGraphics;
        uint               clockMemory;
        uint               clockMemoryMax;
        nvmlPstates_t      performanceState;
    } stats;

    std::vector<std::string> names_ = { "timestamp",
                                        "temperature_gpu",
                                        "power_draw_w",
                                        "power_limit_w",
                                        "utilization_gpu",
                                        "utilization_memory",
                                        "memory_used_mib",
                                        "memory_free_mib",
                                        "clocks_throttle_reasons_active",
                                        "clocks_current_sm_mhz",
                                        "clocks_applications_graphics_mhz",
                                        "clocks_current_memory_mhz",
                                        "clocks_max_memory_mhz",
                                        "pstate" };

    std::vector<stats> time_steps_;
    std::string        filename_;
    std::ofstream      outfile_;
    nvmlDevice_t       device_;
    bool               loop_;

    void printHeader();
    void dumpData();
};
}
