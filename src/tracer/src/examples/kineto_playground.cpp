// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <libkineto.h>
#include "kineto_playground.cuh"

static const std::string kFileName = "/home/suraj/Work/llnl/nvidia-data-movement-experiments/data/kineto-basic-playground/perf.json";

int main() {
  // kineto::warmup();

  // Kineto config
  std::set<libkineto::ActivityType> types = {
      libkineto::ActivityType::CONCURRENT_KERNEL,
      libkineto::ActivityType::GPU_MEMCPY,
      libkineto::ActivityType::GPU_MEMSET,
      libkineto::ActivityType::CUDA_RUNTIME,
      libkineto::ActivityType::EXTERNAL_CORRELATION,
  };

  std::string profiler_config = "ACTIVITIES_WARMUP_PERIOD_SECS=5\n "
                                // "CUPTI_PROFILER_METRICS=kineto__cuda_core_flops\n "
                                // "CUPTI_PROFILER_ENABLE_PER_KERNEL=true\n "
                                "ACTIVITIES_DURATION_SECS=0";

  auto &profiler = libkineto::api().activityProfiler();
  libkineto::api().initProfilerIfRegistered();
  auto isActive = profiler.isActive();
  profiler.prepareTrace(types, profiler_config);

  // // Good to warm up after prepareTrace to get cupti initialization to settle
  // kineto::warmup();
  profiler.startTrace();
  std::cout << "Starting memcpy to device." << std::endl;
  kineto::basicMemcpyToDevice();
  std::cout << "Compute" << std::endl;
  kineto::compute();
  std::cout << "Starting memcpy from device" << std::endl;
  kineto::basicMemcpyFromDevice();

  auto trace = profiler.stopTrace();
  std::cout << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.\n";
  trace->save(kFileName);
  return 0;
}
