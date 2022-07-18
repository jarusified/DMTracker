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

  // Empty types set defaults to all types
  std::set<libkineto::ActivityType> types;

  auto& profiler = libkineto::api().activityProfiler();
  libkineto::api().initProfilerIfRegistered();
  profiler.prepareTrace(types);

  // Good to warm up after prepareTrace to get cupti initialization to settle
  kineto::warmup();
  profiler.startTrace();
  kineto::playground();

  auto trace = profiler.stopTrace();
  std::cout << "Stopped and processed trace. Got " << trace->activities()->size() << " activities.";
  trace->save(kFileName);
  return 0;
}
