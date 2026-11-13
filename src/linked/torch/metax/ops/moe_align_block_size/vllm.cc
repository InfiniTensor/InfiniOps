#include "linked/torch/metax/ops/moe_align_block_size/vllm.h"

namespace infini::ops::linked::torch {

template class TorchMoeAlignBlockSize<
    ::infini::ops::linked::torch::metax::VllmMoeAlignBlockSize>;

}  // namespace infini::ops::linked::torch
