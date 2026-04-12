#ifndef INFINI_OPS_EXAMPLES_RUNTIME_API_H_
#define INFINI_OPS_EXAMPLES_RUNTIME_API_H_

#include "device.h"

#ifdef WITH_NVIDIA
#include "nvidia/gemm/kernel.h"
#include "nvidia/gemm/cublaslt.h"
#include "nvidia/runtime_.h"
#elif WITH_ILUVATAR
#include "iluvatar/gemm/cublas.h"
#include "iluvatar/runtime_.h"
#elif WITH_METAX
#include "metax/gemm/mcblas.h"
#include "metax/runtime_.h"
#elif WITH_CAMBRICON
#include "cambricon/gemm/cnblas.h"
#include "cambricon/runtime_.h"
#elif WITH_MOORE
#include "moore/gemm/mublas.h"
#include "moore/runtime_.h"
#elif WITH_ASCEND
#include "ascend/gemm/kernel.h"
#include "ascend/runtime_.h"
#elif WITH_CPU
#include "cpu/gemm/gemm.h"
#include "cpu/runtime_.h"
#else
#error "One `WITH_*` backend must be enabled for the examples."
#endif

namespace infini::ops {

#ifdef WITH_NVIDIA
using DefaultRuntimeUtils = Runtime<Device::Type::kNvidia>;
#elif WITH_ILUVATAR
using DefaultRuntimeUtils = Runtime<Device::Type::kIluvatar>;
#elif WITH_METAX
using DefaultRuntimeUtils = Runtime<Device::Type::kMetax>;
#elif WITH_CAMBRICON
using DefaultRuntimeUtils = Runtime<Device::Type::kCambricon>;
#elif WITH_MOORE
using DefaultRuntimeUtils = Runtime<Device::Type::kMoore>;
#elif WITH_ASCEND
using DefaultRuntimeUtils = Runtime<Device::Type::kAscend>;
#elif WITH_CPU
using DefaultRuntimeUtils = Runtime<Device::Type::kCpu>;
#endif

}  // namespace infini::ops

#endif
