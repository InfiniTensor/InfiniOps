#ifndef INFINI_OPS_METAX_CAT_KERNEL_H_
#define INFINI_OPS_METAX_CAT_KERNEL_H_

#ifdef WITH_TORCH

#include "base/cat.h"

namespace infini::ops {

template <>
class Operator<Cat, Device::Type::kMetax> : public Cat {
 public:
  using Cat::Cat;

  void operator()(const Tensor first_input, std::vector<Tensor> rest_inputs,
                  int64_t dim, Tensor out) const override;
};

}  // namespace infini::ops

#endif

#endif
