# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

def get_libkineto_api_srcs():
    return [
        "lib/ThreadUtil.cpp",
        "lib/libkineto_api.cpp",
    ]

def get_libkineto_cupti_srcs(with_api = True):
    return [
        "lib/CudaDeviceProperties.cpp",
        "lib/CuptiActivityApi.cpp",
        "lib/CuptiActivityPlatform.cpp",
        "lib/CuptiCallbackApi.cpp",
        "lib/CuptiEventApi.cpp",
        "lib/CuptiMetricApi.cpp",
        "lib/CuptiRangeProfiler.cpp",
        "lib/CuptiRangeProfilerApi.cpp",
        "lib/CuptiRangeProfilerConfig.cpp",
        "lib/CuptiNvPerfMetric.cpp",
        "lib/Demangle.cpp",
        "lib/EventProfiler.cpp",
        "lib/EventProfilerController.cpp",
        "lib/WeakSymbols.cpp",
        "lib/cupti_strings.cpp",
    ] + (get_libkineto_cpu_only_srcs(with_api))

def get_libkineto_roctracer_srcs(with_api = True):
    return [
        "lib/RoctracerActivityApi.cpp",
    ] + (get_libkineto_cpu_only_srcs(with_api))

def get_libkineto_cpu_only_srcs(with_api = True):
    return [
        "lib/AbstractConfig.cpp",
        "lib/CuptiActivityProfiler.cpp",
        "lib/ActivityProfilerController.cpp",
        "lib/ActivityProfilerProxy.cpp",
        "lib/ActivityType.cpp",
        "lib/Config.cpp",
        "lib/ConfigLoader.cpp",
        "lib/CuptiActivityApi.cpp",
        "lib/Demangle.cpp",
        "lib/GenericTraceActivity.cpp",
        "lib/ILoggerObserver.cpp",
        "lib/Logger.cpp",
        "lib/init.cpp",
        "lib/output_csv.cpp",
        "lib/output_json.cpp",
    ] + (get_libkineto_api_srcs() if with_api else [])

def get_libkineto_public_headers():
    return [
        "include/AbstractConfig.h",
        "include/ActivityProfilerInterface.h",
        "include/ActivityTraceInterface.h",
        "include/ActivityType.h",
        "include/Config.h",
        "include/ClientInterface.h",
        "include/GenericTraceActivity.h",
        "include/IActivityProfiler.h",
        "include/ILoggerObserver.h",
        "include/ITraceActivity.h",
        "include/TraceSpan.h",
        "include/ThreadUtil.h",
        "include/libkineto.h",
        "include/time_since_epoch.h",
    ]

# kineto code should be updated to not have to
# suppress these warnings.
KINETO_COMPILER_FLAGS = [
    "-fexceptions",
    "-Wno-deprecated-declarations",
    "-Wno-unused-function",
    "-Wno-unused-private-field",
]
