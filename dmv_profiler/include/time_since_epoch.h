#pragma once

#include <chrono>

namespace libdmv {

template <class ClockT>
inline int64_t timeSinceEpoch(const std::chrono::time_point<ClockT> &t) {
  return std::chrono::duration_cast<std::chrono::microseconds>(
             t.time_since_epoch())
      .count();
}

} // namespace libdmv
