#pragma once

#include <atomic>
#include <chrono>
#include <deque>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "ActivityProfilerInterface.h"
#include "ActivityTraceInterface.h"
#include "ActivityType.h"
#include "ClientInterface.h"
#include "GenericTraceActivity.h"
#include "IActivityProfiler.h"
#include "ThreadUtil.h"
#include "TraceSpan.h"

extern "C" {
void suppresslibdmvLogMessages();
void libdmv_init(bool cpuOnly, bool logOnError);
void libdmv_fin(bool cpuOnly, bool logOnError);
}

namespace libdmv {

class Config;
class ConfigLoader;

struct CpuTraceBuffer {
  template <class... Args> void emplace_activity(Args &&...args) {
    activities.emplace_back(
        std::make_unique<GenericTraceActivity>(std::forward<Args>(args)...));
  }

  static GenericTraceActivity &
  toRef(std::unique_ptr<GenericTraceActivity> &ref) {
    return *ref;
  }

  static const GenericTraceActivity &
  toRef(const std::unique_ptr<GenericTraceActivity> &ref) {
    return *ref;
  }

  TraceSpan span{0, 0, "none"};
  int gpuOpCount;
  std::deque<std::unique_ptr<GenericTraceActivity>> activities;
};

using ChildActivityProfilerFactory =
    std::function<std::unique_ptr<IActivityProfiler>()>;

class libdmvApi {
public:
  explicit libdmvApi(ConfigLoader &configLoader)
      : configLoader_(configLoader) {}

  // Called by client that supports tracing API.
  // libdmv can still function without this.
  void registerClient(ClientInterface *client);

  // Called by libdmv on init
  void registerProfiler(std::unique_ptr<ActivityProfilerInterface> profiler) {
    activityProfiler_ = std::move(profiler);
    initClientIfRegistered();
  }

  ActivityProfilerInterface &activityProfiler() {
    libdmv_init(false, true);
    return *activityProfiler_;
  }

  ClientInterface *client() { return client_; }

  void initProfilerIfRegistered() {
    // LOG (INFO) << "Init profiler";
    static std::once_flag once;
    if (activityProfiler_) {
      std::call_once(once, [this] {
        if (!activityProfiler_->isInitialized()) {
          activityProfiler_->init();
          initChildActivityProfilers();
        }
      });
    }
  }

  bool isProfilerInitialized() const {
    return activityProfiler_ && activityProfiler_->isInitialized();
  }

  bool isProfilerRegistered() const { return activityProfiler_ != nullptr; }

  void suppressLogMessages() { suppresslibdmvLogMessages(); }

  // Provides access to profier configuration manaegement
  ConfigLoader &configLoader() { return configLoader_; }

  void registerProfilerFactory(ChildActivityProfilerFactory factory) {
    if (isProfilerInitialized()) {
      activityProfiler_->addChildActivityProfiler(factory());
    } else {
      childProfilerFactories_.push_back(factory);
    }
  }

private:
  void initChildActivityProfilers() {
    if (!isProfilerInitialized()) {
      return;
    }
    for (const auto &factory : childProfilerFactories_) {
      activityProfiler_->addChildActivityProfiler(factory());
    }
    childProfilerFactories_.clear();
  }

  // Client is initialized once both it and libdmv has registered
  void initClientIfRegistered();

  ConfigLoader &configLoader_;
  std::unique_ptr<ActivityProfilerInterface> activityProfiler_{};
  ClientInterface *client_{};
  int32_t clientRegisterThread_{0};

  bool isLoaded_{false};
  std::vector<ChildActivityProfilerFactory> childProfilerFactories_;
};

// Singleton
libdmvApi &api();

} // namespace libdmv
