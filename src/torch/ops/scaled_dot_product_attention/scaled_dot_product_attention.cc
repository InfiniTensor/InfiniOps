#include "torch/ops/scaled_dot_product_attention/scaled_dot_product_attention.h"

#include "torch/instantiate.h"
#include "torch/stream_guard.h"
#include "torch/tensor_.h"

#if defined(WITH_ILUVATAR) && TORCH_VERSION_MAJOR == 2 && \
    TORCH_VERSION_MINOR == 4
#include <ATen/ops/_scaled_dot_product_attention_math.h>

#include <limits>
#endif

namespace infini::ops {

template <Device::Type kDev>
Operator<ScaledDotProductAttention, kDev, 1>::Operator(
    const Tensor query, const Tensor key, const Tensor value,
    const std::optional<Tensor> attn_mask, double dropout_p, bool is_causal,
    const std::optional<double> scale, bool enable_gqa, Tensor out)
    : ScaledDotProductAttention{query,     key,        value,
                                attn_mask, dropout_p,  is_causal,
                                scale,     enable_gqa, out} {}

template <Device::Type kDev>
Operator<ScaledDotProductAttention, kDev, 1>::Operator(const Tensor query,
                                                       const Tensor key,
                                                       const Tensor value,
                                                       Tensor out)
    : Operator{query, key,          value, std::nullopt, 0.0,
               false, std::nullopt, false, out} {}

template <Device::Type kDev>
void Operator<ScaledDotProductAttention, kDev, 1>::operator()(
    const Tensor query, const Tensor key, const Tensor value,
    const std::optional<Tensor> attn_mask, double dropout_p, bool is_causal,
    const std::optional<double> scale, bool enable_gqa, Tensor out) const {
  const TorchStreamGuard<kDev> guard{stream_, device_index_};
  auto at_query =
      ToAtenTensor<kDev>(const_cast<void*>(query.data()), query_shape_,
                         query_strides_, query_type_, device_index_);
  auto at_key = ToAtenTensor<kDev>(const_cast<void*>(key.data()), key_shape_,
                                   key_strides_, query_type_, device_index_);
  auto at_value =
      ToAtenTensor<kDev>(const_cast<void*>(value.data()), value_shape_,
                         value_strides_, query_type_, device_index_);
  auto at_out = ToAtenTensor<kDev>(out.data(), out_shape_, out_strides_,
                                   query_type_, device_index_);

  c10::optional<at::Tensor> at_attn_mask;
  if (attn_mask.has_value()) {
    at_attn_mask.emplace(ToAtenTensor<kDev>(
        const_cast<void*>(attn_mask->data()), attn_mask_shape_,
        attn_mask_strides_, attn_mask_type_, device_index_));
  }

  c10::optional<double> at_scale;
  if (scale.has_value()) {
    at_scale = *scale;
  }

#if TORCH_VERSION_MAJOR > 2 || \
    (TORCH_VERSION_MAJOR == 2 && TORCH_VERSION_MINOR >= 5)
  auto result = at::scaled_dot_product_attention(
      at_query, at_key, at_value, at_attn_mask, dropout_p, is_causal, at_scale,
      enable_gqa);
#else
  if (enable_gqa) {
    at_key = at_key.repeat_interleave(at_query.size(-3) / at_key.size(-3), -3);
    at_value =
        at_value.repeat_interleave(at_query.size(-3) / at_value.size(-3), -3);
  }

#if defined(WITH_ILUVATAR) && TORCH_VERSION_MAJOR == 2 && \
    TORCH_VERSION_MINOR == 4
  if constexpr (kDev == Device::Type::kIluvatar) {
    // CoreX Torch 2.4 fused SDPA does not support FP32 or unequal query/value
    // head widths. Use math without changing global backend flags.
    if (query_type_ == DataType::kFloat32 ||
        query_shape_.back() != value_shape_.back()) {
      auto math_mask = at_attn_mask;
      if (math_mask.has_value() && math_mask->scalar_type() == at::kBool) {
        math_mask = at::zeros_like(*math_mask, at_query.options())
                        .masked_fill_(math_mask->logical_not(),
                                      -std::numeric_limits<float>::infinity());
      }
      auto result = std::get<0>(at::_scaled_dot_product_attention_math(
          at_query, at_key, at_value, math_mask, dropout_p, is_causal,
          c10::nullopt, at_scale));
      at_out.copy_(result);
      return;
    }
  }
#endif

  auto result = at::scaled_dot_product_attention(
      at_query, at_key, at_value, at_attn_mask, dropout_p, is_causal, at_scale);
#endif
  at_out.copy_(result);
}

INFINI_OPS_INSTANTIATE_TORCH_OPERATOR(ScaledDotProductAttention, 1)

}  // namespace infini::ops
