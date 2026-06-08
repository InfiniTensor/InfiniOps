#ifndef INFINI_OPS_CAMBRICON_CAUSAL_SOFTMAX_KERNEL_H_
#define INFINI_OPS_CAMBRICON_CAUSAL_SOFTMAX_KERNEL_H_

#ifdef WITH_TORCH

#include <limits>

#include "base/causal_softmax.h"
#include "native/cambricon/ops/torch_fallback.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmax, Device::Type::kCambricon> : public CausalSoftmax {
 public:
  using CausalSoftmax::CausalSoftmax;

  void operator()(const Tensor input, Tensor out) const override {
    auto at_input = cambricon_torch_fallback::ToAten(input);
    auto at_out = cambricon_torch_fallback::ToAten(out);

    auto run = [&](at::Tensor work_input) {
      auto work = work_input.to(at::kFloat);
      auto index_options =
          at::TensorOptions().device(work.device()).dtype(at::kLong);
      auto rows = at::arange(static_cast<int64_t>(seq_len_), index_options)
                      .unsqueeze(-1);
      auto cols =
          at::arange(static_cast<int64_t>(total_seq_len_), index_options)
              .unsqueeze(0);
      auto first_valid_col =
          rows + static_cast<int64_t>(total_seq_len_ - seq_len_ + 1);
      auto mask = cols >= first_valid_col;

      if (ndim_ == 3) {
        mask = mask.unsqueeze(0);
      }

      auto masked =
          work.masked_fill(mask, -std::numeric_limits<float>::infinity());
      auto result = at::softmax(masked, -1);
      cambricon_torch_fallback::CopyToOutput(at_out, std::move(result));
    };

    try {
      run(at_input);
    } catch (const c10::Error&) {
      run(at_input.cpu());
    }
  }
};

}  // namespace infini::ops

#endif

#endif
