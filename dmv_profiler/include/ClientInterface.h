#pragma once

namespace libdmv {

class ClientInterface {
public:
  virtual ~ClientInterface() {}
  virtual void init() = 0;
  virtual void warmup(bool setupOpInputsCollection) = 0;
  virtual void start() = 0;
  virtual void stop() = 0;
};

} // namespace libdmv
