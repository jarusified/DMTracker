#include <string>
#include <iostream>

#include "profiler.h"
#include "main.cuh"


int main() {
  dmv_profiler_wrapper::init_profile();
  
  kineto::warmup();
  if(dmv_profiler_wrapper::is_profiling()) {
    dmv_profiler_wrapper::start_profile();

    std::cout << "Starting memcpy to device." << std::endl;
    kineto::basicMemcpyToDevice();
    std::cout << "Compute" << std::endl;
    kineto::compute();
    std::cout << "Starting memcpy from device" << std::endl;
    kineto::basicMemcpyFromDevice();

    dmv_profiler_wrapper::stop_profile();
    dmv_profiler_wrapper::save_profile();
  }
  
  return 0;
}
