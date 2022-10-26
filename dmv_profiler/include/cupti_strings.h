#pragma once

#include <cupti.h>

namespace libdmv {

const char *memoryKindString(CUpti_ActivityMemoryKind kind);
const char *memcpyKindString(CUpti_ActivityMemcpyKind kind);
const char *runtimeCbidName(CUpti_CallbackId cbid);
const char *overheadKindString(CUpti_ActivityOverheadKind kind);

} // namespace libdmv
