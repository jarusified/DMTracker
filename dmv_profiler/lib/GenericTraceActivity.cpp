#include "GenericTraceActivity.h"
#include "output_base.h"

namespace libdmv {
void GenericTraceActivity::log(ActivityLogger &logger) const {
  logger.handleGenericActivity(*this);
}
} // namespace libdmv
