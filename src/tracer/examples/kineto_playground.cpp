// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <string>
#include <iostream>
#include <filesystem>

#include "libkineto.h"
#include "kineto_playground.cuh"

namespace fs = std::filesystem;

int main() {
  std::string kFileName = "kineto-basic-playground_perf.json";

  kineto::warmup();

  // Kineto config
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
  auto isActive = profiler.isActive();

  // Good to warm up after prepareTrace to get cupti initialization to settle
  if(isActive) {
    kineto::warmup();
    profiler.startTrace();
  
    std::cout << "Starting memcpy to device." << std::endl;
    kineto::basicMemcpyToDevice();
    std::cout << "Compute" << std::endl;
    kineto::compute();
    std::cout << "Starting memcpy from device" << std::endl;
    kineto::basicMemcpyFromDevice();

    auto trace = profiler.stopTrace();
    std::cout << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.\n";
    std::string currDirPath = fs::current_path();
    std::string filePath = currDirPath + "/" + kFileName;
    trace->save(filePath);
  }
  return 0;
}
