#ifndef INFINI_OPS_TORCH_INSTANTIATE_H_
#define INFINI_OPS_TORCH_INSTANTIATE_H_

// Emit definitions only for the backends enabled in this translation unit.

#ifdef WITH_CPU
#define INFINI_OPS_TORCH_INSTANTIATE_CPU(Op, Index) \
  template class Operator<Op, Device::Type::kCpu, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_CPU(Op, Index)
#endif

#ifdef WITH_NVIDIA
#define INFINI_OPS_TORCH_INSTANTIATE_NVIDIA(Op, Index) \
  template class Operator<Op, Device::Type::kNvidia, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_NVIDIA(Op, Index)
#endif

#ifdef WITH_CAMBRICON
#define INFINI_OPS_TORCH_INSTANTIATE_CAMBRICON(Op, Index) \
  template class Operator<Op, Device::Type::kCambricon, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_CAMBRICON(Op, Index)
#endif

#ifdef WITH_ASCEND
#define INFINI_OPS_TORCH_INSTANTIATE_ASCEND(Op, Index) \
  template class Operator<Op, Device::Type::kAscend, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_ASCEND(Op, Index)
#endif

#ifdef WITH_METAX
#define INFINI_OPS_TORCH_INSTANTIATE_METAX(Op, Index) \
  template class Operator<Op, Device::Type::kMetax, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_METAX(Op, Index)
#endif

#ifdef WITH_MOORE
#define INFINI_OPS_TORCH_INSTANTIATE_MOORE(Op, Index) \
  template class Operator<Op, Device::Type::kMoore, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_MOORE(Op, Index)
#endif

#ifdef WITH_ILUVATAR
#define INFINI_OPS_TORCH_INSTANTIATE_ILUVATAR(Op, Index) \
  template class Operator<Op, Device::Type::kIluvatar, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_ILUVATAR(Op, Index)
#endif

#ifdef WITH_HYGON
#define INFINI_OPS_TORCH_INSTANTIATE_HYGON(Op, Index) \
  template class Operator<Op, Device::Type::kHygon, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_HYGON(Op, Index)
#endif

#ifdef WITH_THEAD
#define INFINI_OPS_TORCH_INSTANTIATE_THEAD(Op, Index) \
  template class Operator<Op, Device::Type::kThead, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_THEAD(Op, Index)
#endif

#ifdef WITH_MARS
#define INFINI_OPS_TORCH_INSTANTIATE_MARS(Op, Index) \
  template class Operator<Op, Device::Type::kMars, Index>;
#else
#define INFINI_OPS_TORCH_INSTANTIATE_MARS(Op, Index)
#endif

#define INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(Op, Index) \
  INFINI_OPS_TORCH_INSTANTIATE_CPU(Op, Index)            \
  INFINI_OPS_TORCH_INSTANTIATE_NVIDIA(Op, Index)         \
  INFINI_OPS_TORCH_INSTANTIATE_CAMBRICON(Op, Index)      \
  INFINI_OPS_TORCH_INSTANTIATE_ASCEND(Op, Index)         \
  INFINI_OPS_TORCH_INSTANTIATE_METAX(Op, Index)          \
  INFINI_OPS_TORCH_INSTANTIATE_MOORE(Op, Index)          \
  INFINI_OPS_TORCH_INSTANTIATE_ILUVATAR(Op, Index)       \
  INFINI_OPS_TORCH_INSTANTIATE_HYGON(Op, Index)          \
  INFINI_OPS_TORCH_INSTANTIATE_THEAD(Op, Index)          \
  INFINI_OPS_TORCH_INSTANTIATE_MARS(Op, Index)

#endif  // INFINI_OPS_TORCH_INSTANTIATE_H_
