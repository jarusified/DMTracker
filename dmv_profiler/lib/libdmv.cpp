#include "libdmv.h"

#include "ConfigLoader.h"
#include "ThreadUtil.h"

namespace libdmv {

libdmvApi &api() {
  static libdmvApi instance(ConfigLoader::instance());
  return instance;
}

void libdmvApi::initClientIfRegistered() {
  if (client_) {
    if (clientRegisterThread_ != threadId()) {
      fprintf(stderr,
              "ERROR: External init callback must run in same thread as "
              "registerClient "
              "(%d != %d)\n",
              threadId(), (int)clientRegisterThread_);
    } else {
      client_->init();
    }
  }
}

void libdmvApi::registerClient(ClientInterface *client) {
  client_ = client;
  if (client && activityProfiler_) {
    // Can initialize straight away
    client->init();
  }
  // Assume here that the external init callback is *not* threadsafe
  // and only call it if it's the same thread that called registerClient
  clientRegisterThread_ = threadId();
}

} // namespace libdmv
