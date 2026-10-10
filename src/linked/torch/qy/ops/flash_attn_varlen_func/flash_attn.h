#ifndef INFINI_OPS_LINKED_TORCH_QY_OPS_FLASH_ATTN_VARLEN_FUNC_FLASH_ATTN_H_
#define INFINI_OPS_LINKED_TORCH_QY_OPS_FLASH_ATTN_VARLEN_FUNC_FLASH_ATTN_H_

#include <ATen/core/Generator.h>

#include "linked/torch/ops/flash_attn_varlen_func.h"
#include "torch/qy/c10.h"

namespace infini::ops::linked::torch::qy {

struct FlashAttnVarlen : C10<Device::Type::kQy> {
  static std::vector<at::Tensor> Call(
      at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
      std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
      std::optional<const at::Tensor>& leftpad_k,
      std::optional<at::Tensor>& block_table,
      std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q,
      int max_seqlen_k, float dropout_p, float softmax_scale, bool zero_tensors,
      bool causal, int window_size_left, int window_size_right, float softcap,
      bool return_softmax, std::optional<at::Generator> generator);
};

}  // namespace infini::ops::linked::torch::qy

namespace infini::ops::linked::torch {

extern template class TorchFlashAttnVarlenFunc<
    ::infini::ops::linked::torch::qy::FlashAttnVarlen>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<FlashAttnVarlenFunc, Device::Type::kQy, 16>
    : public linked::torch::TorchFlashAttnVarlenFunc<
          linked::torch::qy::FlashAttnVarlen> {
 public:
  using linked::torch::TorchFlashAttnVarlenFunc<
      linked::torch::qy::FlashAttnVarlen>::TorchFlashAttnVarlenFunc;

  using linked::torch::TorchFlashAttnVarlenFunc<
      linked::torch::qy::FlashAttnVarlen>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_QY_OPS_FLASH_ATTN_VARLEN_FUNC_FLASH_ATTN_H_
