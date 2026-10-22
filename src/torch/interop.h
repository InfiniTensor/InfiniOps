#ifndef INFINI_OPS_TORCH_INTEROP_H_
#define INFINI_OPS_TORCH_INTEROP_H_

// Backend assembly is the only shared header listing platform adapters.
#include "torch/cpu/c10.h"
#include "torch/interop_base.h"
#ifdef WITH_NVIDIA
#include "torch/nvidia/c10.h"
#endif
#ifdef WITH_CAMBRICON
#include "torch/cambricon/c10.h"
#endif
#ifdef WITH_ASCEND
#include "torch/ascend/interop.h"
#endif
#ifdef WITH_METAX
#include "torch/metax/c10.h"
#endif
#ifdef WITH_MOORE
#include "torch/moore/c10.h"
#endif
#ifdef WITH_ILUVATAR
#include "torch/iluvatar/c10.h"
#endif
#ifdef WITH_HYGON
#include "torch/hygon/c10.h"
#endif
#ifdef WITH_THEAD
#include "torch/thead/c10.h"
#endif
#ifdef WITH_MARS
#include "torch/mars/c10.h"
#endif

#endif  // INFINI_OPS_TORCH_INTEROP_H_
