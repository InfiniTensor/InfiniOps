#include "linked/torch/moore/ops/flash_attn_with_kvcache/mate.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "linked/torch/moore/mate.h"
#include "linked/torch/ops/flash_attn_with_kvcache.h"
#include "torch/moore/c10.h"
namespace infini::ops::linked::torch::moore {
namespace {

std::vector<at::Tensor> MateFlashAttnKvcacheKernel(
    at::Tensor q, at::Tensor k_cache, at::Tensor v_cache,
    std::optional<at::Tensor> k, std::optional<at::Tensor> v,
    std::optional<at::Tensor> cache_seqlens,
    std::optional<at::Tensor> rotary_cos, std::optional<at::Tensor> rotary_sin,
    std::optional<at::Tensor> cache_batch_idx,
    std::optional<at::Tensor> cache_leftpad,
    std::optional<at::Tensor> block_table, double softmax_scale, bool causal,
    int64_t window_size_left, int64_t window_size_right, double softcap,
    bool rotary_interleaved, int64_t num_splits) {
  namespace py = pybind11;
  py::gil_scoped_acquire gil;

  try {
    auto mha_interface = detail::ImportMateMhaInterface();
    auto result = mha_interface.attr("flash_attn_with_kvcache")(
        py::arg("q") = q, py::arg("k_cache") = k_cache,
        py::arg("v_cache") = v_cache,
        py::arg("k") = detail::OptionalTensorToPyObject(k),
        py::arg("v") = detail::OptionalTensorToPyObject(v),
        py::arg("rotary_cos") = detail::OptionalTensorToPyObject(rotary_cos),
        py::arg("rotary_sin") = detail::OptionalTensorToPyObject(rotary_sin),
        py::arg("cache_seqlens") =
            detail::OptionalTensorToPyObject(cache_seqlens),
        py::arg("cache_batch_idx") =
            detail::OptionalTensorToPyObject(cache_batch_idx),
        py::arg("cache_leftpad") =
            detail::OptionalTensorToPyObject(cache_leftpad),
        py::arg("page_table") = detail::OptionalTensorToPyObject(block_table),
        py::arg("softmax_scale") = softmax_scale, py::arg("causal") = causal,
        py::arg("window_size") =
            py::make_tuple(window_size_left, window_size_right),
        py::arg("softcap") = softcap,
        py::arg("rotary_interleaved") = rotary_interleaved,
        py::arg("num_splits") = num_splits,
        py::arg("pack_gqa") = (q.size(-2) != k_cache.size(-2)),
        py::arg("return_softmax_lse") = true);

    auto outputs = result.cast<py::tuple>();
    TORCH_CHECK(outputs.size() >= 2,
                "Mate flash_attn_with_kvcache returned no softmax LSE");
    return {outputs[0].cast<at::Tensor>(), outputs[1].cast<at::Tensor>()};
  } catch (const py::error_already_set& error) {
    TORCH_CHECK(false, "Mate flash_attn_with_kvcache failed: ", error.what());
  }
}

}  // namespace
}  // namespace infini::ops::linked::torch::moore

TORCH_LIBRARY_FRAGMENT(infini_moore_mate, m) {
  m.def(
      "flash_attn_with_kvcache(Tensor q, Tensor k_cache, Tensor v_cache, "
      "Tensor? k, Tensor? v, Tensor? cache_seqlens, Tensor? rotary_cos, "
      "Tensor? rotary_sin, Tensor? cache_batch_idx, Tensor? cache_leftpad, "
      "Tensor? block_table, float softmax_scale, bool causal, "
      "int window_size_left, int window_size_right, float softcap, "
      "bool rotary_interleaved, int num_splits) -> Tensor[]");
}

TORCH_LIBRARY_IMPL(infini_moore_mate, PrivateUse1, m) {
  m.impl(
      "flash_attn_with_kvcache",
      TORCH_FN(infini::ops::linked::torch::moore::MateFlashAttnKvcacheKernel));
}

namespace infini::ops::linked::torch::moore {

struct MateFlashAttnKvcache : C10<Device::Type::kMoore> {
  static std::vector<at::Tensor> Call(
      at::Tensor& q, const at::Tensor& k_cache, const at::Tensor& v_cache,
      std::optional<const at::Tensor>& k, std::optional<const at::Tensor>& v,
      std::optional<const at::Tensor>& cache_seqlens,
      std::optional<const at::Tensor>& rotary_cos,
      std::optional<const at::Tensor>& rotary_sin,
      std::optional<const at::Tensor>& cache_batch_idx,
      std::optional<const at::Tensor>& cache_leftpad,
      std::optional<at::Tensor>& block_table,
      std::optional<at::Tensor>& alibi_slopes, std::optional<at::Tensor>& out,
      float softmax_scale, bool causal, int window_size_left,
      int window_size_right, float softcap, bool rotary_interleaved,
      int num_splits) {
    TORCH_CHECK(!alibi_slopes.has_value(), "Mate ALiBi is not supported");
    TORCH_CHECK(!out.has_value(), "Mate does not accept an output tensor here");

    auto mutable_optional = [](const std::optional<const at::Tensor>& tensor) {
      return tensor.has_value() ? std::optional<at::Tensor>(*tensor)
                                : std::nullopt;
    };

    static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "infini_moore_mate::flash_attn_with_kvcache", "");
    c10::Stack stack;
    stack.emplace_back(q);
    stack.emplace_back(k_cache);
    stack.emplace_back(v_cache);
    const auto at_k = mutable_optional(k);
    const auto at_v = mutable_optional(v);
    const auto at_cache_seqlens = mutable_optional(cache_seqlens);
    const auto at_rotary_cos = mutable_optional(rotary_cos);
    const auto at_rotary_sin = mutable_optional(rotary_sin);
    const auto at_cache_batch_idx = mutable_optional(cache_batch_idx);
    const auto at_cache_leftpad = mutable_optional(cache_leftpad);
    stack.emplace_back(at_k.has_value() ? c10::IValue(*at_k) : c10::IValue());
    stack.emplace_back(at_v.has_value() ? c10::IValue(*at_v) : c10::IValue());
    stack.emplace_back(at_cache_seqlens.has_value()
                           ? c10::IValue(*at_cache_seqlens)
                           : c10::IValue());
    stack.emplace_back(at_rotary_cos.has_value() ? c10::IValue(*at_rotary_cos)
                                                 : c10::IValue());
    stack.emplace_back(at_rotary_sin.has_value() ? c10::IValue(*at_rotary_sin)
                                                 : c10::IValue());
    stack.emplace_back(at_cache_batch_idx.has_value()
                           ? c10::IValue(*at_cache_batch_idx)
                           : c10::IValue());
    stack.emplace_back(at_cache_leftpad.has_value()
                           ? c10::IValue(*at_cache_leftpad)
                           : c10::IValue());
    stack.emplace_back(block_table.has_value() ? c10::IValue(*block_table)
                                               : c10::IValue());
    stack.emplace_back(static_cast<double>(softmax_scale));
    stack.emplace_back(causal);
    stack.emplace_back(static_cast<int64_t>(window_size_left));
    stack.emplace_back(static_cast<int64_t>(window_size_right));
    stack.emplace_back(static_cast<double>(softcap));
    stack.emplace_back(rotary_interleaved);
    stack.emplace_back(static_cast<int64_t>(num_splits));
    op.callBoxed(&stack);

    TORCH_CHECK(stack.size() == 1,
                "Mate flash_attn_with_kvcache returned an invalid result");
    return stack.front().toTensorVector();
  }
};

}  // namespace infini::ops::linked::torch::moore

namespace infini::ops {

void Operator<FlashAttnWithKvcache, Device::Type::kMoore, 16>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin, const int64_t cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  using Delegate = linked::torch::TorchFlashAttnWithKvcache<
      linked::torch::moore::MateFlashAttnKvcache>;
  if (!delegate_) {
    delegate_ = std::make_unique<Delegate>(
        q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
        cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
        softmax_scale, causal, window_size, softcap, rotary_interleaved,
        num_splits, return_softmax_lse, out, softmax_lse);
  }
  delegate_->set_stream(stream_);
  (*delegate_)(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
               cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
               softmax_scale, causal, window_size, softcap, rotary_interleaved,
               num_splits, return_softmax_lse, out, softmax_lse);
}

void Operator<FlashAttnWithKvcache, Device::Type::kMoore, 16>::operator()(
    const Tensor q, Tensor k_cache, Tensor v_cache,
    const std::optional<Tensor> k, const std::optional<Tensor> v,
    const std::optional<Tensor> rotary_cos,
    const std::optional<Tensor> rotary_sin,
    const std::optional<Tensor> cache_seqlens,
    const std::optional<Tensor> cache_batch_idx,
    const std::optional<Tensor> cache_leftpad,
    const std::optional<Tensor> block_table,
    const std::optional<Tensor> alibi_slopes,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool rotary_interleaved, const int64_t num_splits,
    const bool return_softmax_lse, Tensor out,
    std::optional<Tensor> softmax_lse) const {
  using Delegate = linked::torch::TorchFlashAttnWithKvcache<
      linked::torch::moore::MateFlashAttnKvcache>;
  if (!delegate_) {
    delegate_ = std::make_unique<Delegate>(
        q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
        cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
        softmax_scale, causal, window_size, softcap, rotary_interleaved,
        num_splits, return_softmax_lse, out, softmax_lse);
  }
  delegate_->set_stream(stream_);
  (*delegate_)(q, k_cache, v_cache, k, v, rotary_cos, rotary_sin, cache_seqlens,
               cache_batch_idx, cache_leftpad, block_table, alibi_slopes,
               softmax_scale, causal, window_size, softcap, rotary_interleaved,
               num_splits, return_softmax_lse, out, softmax_lse);
}

}  // namespace infini::ops
