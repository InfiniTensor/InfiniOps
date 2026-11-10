#include "linked/torch/metax/ops/layer_norm/aten.h"

namespace infini::ops::linked::torch {

template class TorchLayerNorm<
    ::infini::ops::linked::torch::metax::AtenLayerNorm>;

}  // namespace infini::ops::linked::torch
