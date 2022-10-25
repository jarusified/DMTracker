#pragma once

#include <memory>
#include <string>
#include <vector>

namespace libdmv {

struct ITraceActivity;

class ActivityTraceInterface {
 public:
  virtual ~ActivityTraceInterface() {}
  virtual const std::vector<const ITraceActivity*>* activities() {
    return nullptr;
  }
  virtual void save(const std::string& path) {}
};

} // namespace libdmv
