// Copyright (c) Meta Platforms, Inc. and affiliates.

// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <string>

#include <libkineto.h>
#include "kineto_playground.cuh"

static const std::string kFileName = "/tmp/kineto_playground_trace.json";

int main() {
  kineto::warmup();

  // Kineto config
 std::set<libkineto::ActivityType> types = {
    libkineto::ActivityType::CONCURRENT_KERNEL,
    libkineto::ActivityType::GPU_MEMCPY,
    libkineto::ActivityType::GPU_MEMSET,
    libkineto::ActivityType::CUDA_RUNTIME,
    libkineto::ActivityType::EXTERNAL_CORRELATION,
    libkineto::ActivityType::OVERHEAD
  };
  auto& profiler = libkineto::api().activityProfiler();
  libkineto::api().initProfilerIfRegistered();
  auto isActive = profiler.isActive();
  profiler.prepareTrace(types);

  // // Good to warm up after prepareTrace to get cupti initialization to settle
  kineto::warmup();
  profiler.startTrace();
  kineto::playground();

  auto trace = profiler.stopTrace();
  std::cout << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.";
  trace->save(kFileName);
  return 0;
}
