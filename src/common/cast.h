#ifndef INFINI_OPS_COMMON_CAST_H_
#define INFINI_OPS_COMMON_CAST_H_

#if defined(WITH_NVIDIA) || defined(WITH_ILUVATAR) || defined(WITH_METAX) || \
    defined(WITH_MOORE)
#include "common/cuda/cast.h"
#else
#include "common/cpu/cast.h"
#endif

#endif
