#pragma once
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>

#include "operator.h"

namespace infini::ops {
class CausalConv1d : public Operator<CausalConv1d> {
 public:
  CausalConv1d(const Tensor qkv, const Tensor weight, Tensor conv_state,
               const std::optional<Tensor> bias,
               const std::optional<Tensor> cu_seqlens,
               const std::optional<Tensor> initial_state_indices,
               const std::optional<Tensor> final_state_indices, Tensor out,
               std::optional<Tensor> final_conv_state)
      : metadata_{qkv,
                  weight,
                  conv_state,
                  bias,
                  cu_seqlens,
                  initial_state_indices,
                  final_state_indices,
                  out,
                  final_conv_state} {
    const auto& views = metadata_;
    Validate(views);
    device_ = qkv.device();
    dtype_ = qkv.dtype();
    tokens_ = qkv.shape()[1];
    channels_ = qkv.shape()[2];
    pool_ = conv_state.shape()[0];
    requests_ = cu_seqlens ? cu_seqlens->numel() - 1 : qkv.shape()[0];
  }
  virtual void operator()(const Tensor qkv, const Tensor weight,
                          Tensor conv_state, const std::optional<Tensor> bias,
                          const std::optional<Tensor> cu_seqlens,
                          const std::optional<Tensor> initial_state_indices,
                          const std::optional<Tensor> final_state_indices,
                          Tensor out,
                          std::optional<Tensor> final_conv_state) const = 0;

 protected:
  using Views = std::array<std::optional<Tensor>, 9>;
  Device device_;
  DataType dtype_;
  size_t tokens_, channels_, pool_, requests_;
  const Views metadata_;
  static void Validate(const Views& v) {
    for (const auto& view : v)
      assert((!view || view->shape().size() == view->strides().size()) &&
             "causal conv shape/stride rank mismatch");
    const auto &q = *v[0], &w = *v[1], &state = *v[2], &out = *v[7];
    const auto dtype = q.dtype();
    assert((dtype == DataType::kFloat32 || dtype == DataType::kFloat16 ||
            dtype == DataType::kBFloat16) &&
           "causal conv supports F16/BF16/F32");
    assert(q.ndim() == 3 && w.ndim() == 3 && state.ndim() == 3 &&
           out.shape() == q.shape() && "causal conv rank/shape mismatch");
    const size_t channels = q.shape()[2];
    assert(w.shape()[0] == channels && w.shape()[1] == 1 && w.shape()[2] > 0 &&
           state.shape()[1] == channels &&
           state.shape()[2] == w.shape()[2] - 1 &&
           "causal conv weight/state shape mismatch");
    assert(v[6].has_value() != v[8].has_value() &&
           "causal conv requires exactly one final state destination mode");
    size_t requests = q.shape()[0];
    if (v[4]) {
      assert(q.shape()[0] == 1 && v[4]->ndim() == 1 && v[4]->numel() >= 2 &&
             "causal conv packed sequence shape mismatch");
      requests = v[4]->numel() - 1;
    }
    assert((v[5] || v[6] || state.shape()[0] == requests) &&
           "causal conv state batch mismatch");
    assert((!v[8] || v[8]->shape() ==
                         Tensor::Shape{requests, channels, state.shape()[2]}) &&
           "causal conv final state shape mismatch");
    assert((!v[3] || v[3]->shape() == Tensor::Shape{channels}) &&
           "causal conv bias shape mismatch");
    for (size_t i = 0; i < v.size(); ++i) {
      if (!v[i]) continue;
      assert(v[i]->device() == q.device() && "causal conv device mismatch");
      if (i >= 4 && i <= 6) {
        assert(v[i]->ndim() == 1 && v[i]->IsContiguous() &&
               (v[i]->dtype() == DataType::kInt32 ||
                v[i]->dtype() == DataType::kInt64) &&
               (i == 4 || v[i]->numel() == requests) &&
               "causal conv requires contiguous I32/I64 sequence/slot indices");
      } else {
        assert(v[i]->dtype() == dtype && "causal conv data dtype mismatch");
      }
      // Preserve ordinary positive strided data and reject broadcast
      // destinations.
      if (i == 2 || i == 7 || i == 8)
        for (size_t axis = 0; axis < v[i]->ndim(); ++axis)
          assert((v[i]->shape()[axis] == 1 || v[i]->strides()[axis] != 0) &&
                 "causal conv broadcast destination unsupported");
    }
    if (q.shape()[1] >
            static_cast<size_t>(std::numeric_limits<int64_t>::max()) ||
        state.shape()[0] >
            static_cast<size_t>(std::numeric_limits<int64_t>::max()))
      throw std::overflow_error("causal conv index extent exceeds int64");
  }
};
}  // namespace infini::ops
