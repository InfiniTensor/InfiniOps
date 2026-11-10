#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_LAYER_NORM_ATEN_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_LAYER_NORM_ATEN_H_

#include <ATen/ops/native_layer_norm.h>

#include "linked/torch/ops/layer_norm.h"
#include "torch/metax/c10.h"

namespace infini::ops::linked::torch::metax {

struct AtenLayerNorm : C10<Device::Type::kMetax> {};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

extern template class TorchLayerNorm<
    ::infini::ops::linked::torch::metax::AtenLayerNorm>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<LayerNorm, Device::Type::kMetax, 0>
    : public linked::torch::TorchLayerNorm<
          linked::torch::metax::AtenLayerNorm> {
 public:
  using linked::torch::TorchLayerNorm<
      linked::torch::metax::AtenLayerNorm>::TorchLayerNorm;
  using linked::torch::TorchLayerNorm<
      linked::torch::metax::AtenLayerNorm>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_LAYER_NORM_ATEN_H_
