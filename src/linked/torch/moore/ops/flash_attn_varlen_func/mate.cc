#include "linked/torch/moore/ops/flash_attn_varlen_func/mate.h"

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <torch/library.h>

#include "linked/torch/moore/mate.h"
#include "linked/torch/ops/flash_attn_varlen_func.h"
#include "torch/moore/c10.h"
namespace infini::ops::linked::torch::moore {
namespace {

std::vector<at::Tensor> MateFlashAttnVarlenKernel(
    at::Tensor q, at::Tensor k, at::Tensor v, at::Tensor cu_seqlens_q,
    at::Tensor cu_seqlens_k, std::optional<at::Tensor> block_table,
    int64_t max_seqlen_q, int64_t max_seqlen_k, double softmax_scale,
    bool causal, int64_t window_size_left, int64_t window_size_right,
    double softcap) {
  namespace py = pybind11;
  py::gil_scoped_acquire gil;

  try {
    auto mha_interface = detail::ImportMateMhaInterface();
    py::object mate_cu_seqlens_k = py::cast(cu_seqlens_k);
    py::object seqused_k = py::none();
    if (block_table.has_value()) {
      seqused_k =
          py::cast(cu_seqlens_k.slice(0, 1) - cu_seqlens_k.slice(0, 0, -1));
      mate_cu_seqlens_k = py::none();
    }

    auto result = mha_interface.attr("flash_attn_varlen_func")(
        py::arg("q") = q, py::arg("k") = k, py::arg("v") = v,
        py::arg("cu_seqlens_q") = cu_seqlens_q,
        py::arg("cu_seqlens_k") = mate_cu_seqlens_k,
        py::arg("seqused_k") = seqused_k,
        py::arg("max_seqlen_q") = max_seqlen_q,
        py::arg("max_seqlen_k") = max_seqlen_k,
        py::arg("page_table") = detail::OptionalTensorToPyObject(block_table),
        py::arg("softmax_scale") = softmax_scale, py::arg("causal") = causal,
        py::arg("window_size") =
            py::make_tuple(window_size_left, window_size_right),
        py::arg("softcap") = softcap, py::arg("num_splits") = 0,
        py::arg("pack_gqa") = (q.size(-2) != k.size(-2)),
        py::arg("deterministic") = false, py::arg("return_softmax_lse") = true,
        py::arg("backend") = "auto");

    auto outputs = result.cast<py::tuple>();
    TORCH_CHECK(outputs.size() >= 2,
                "Mate flash_attn_varlen_func returned no softmax LSE");
    return {outputs[0].cast<at::Tensor>(), outputs[1].cast<at::Tensor>(),
            at::empty({0}, q.options())};
  } catch (const py::error_already_set& error) {
    TORCH_CHECK(false, "Mate flash_attn_varlen_func failed: ", error.what());
  }
}

}  // namespace
}  // namespace infini::ops::linked::torch::moore

TORCH_LIBRARY_FRAGMENT(infini_moore_mate, m) {
  m.def(
      "flash_attn_varlen_func(Tensor q, Tensor k, Tensor v, "
      "Tensor cu_seqlens_q, Tensor cu_seqlens_k, Tensor? block_table, "
      "int max_seqlen_q, int max_seqlen_k, float softmax_scale, bool causal, "
      "int window_size_left, int window_size_right, float softcap) -> "
      "Tensor[]");
}

TORCH_LIBRARY_IMPL(infini_moore_mate, PrivateUse1, m) {
  m.impl(
      "flash_attn_varlen_func",
      TORCH_FN(infini::ops::linked::torch::moore::MateFlashAttnVarlenKernel));
}

namespace infini::ops::linked::torch::moore {

struct MateFlashAttnVarlen : C10<Device::Type::kMoore> {
  static std::vector<at::Tensor> Call(
      at::Tensor& q, const at::Tensor& k, const at::Tensor& v,
      std::optional<at::Tensor>& out, const at::Tensor& cu_seqlens_q,
      const at::Tensor& cu_seqlens_k, std::optional<at::Tensor>& seqused_k,
      std::optional<const at::Tensor>& leftpad_k,
      std::optional<at::Tensor>& block_table,
      std::optional<at::Tensor>& alibi_slopes, int max_seqlen_q,
      int max_seqlen_k, float dropout_p, float softmax_scale, bool zero_tensors,
      bool causal, int window_size_left, int window_size_right, float softcap,
      bool return_softmax, std::optional<at::Generator> generator) {
    TORCH_CHECK(!out.has_value(), "Mate does not accept an output tensor here");
    TORCH_CHECK(!seqused_k.has_value(), "Mate seqused_k is not supported yet");
    TORCH_CHECK(!leftpad_k.has_value(), "Mate leftpad_k is not supported yet");
    TORCH_CHECK(!alibi_slopes.has_value(), "Mate ALiBi is not supported");
    TORCH_CHECK(dropout_p == 0.0f, "Mate attention dropout is not supported");
    TORCH_CHECK(!generator.has_value(),
                "Mate attention generators are not supported");
    (void)zero_tensors;
    (void)return_softmax;

    static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "infini_moore_mate::flash_attn_varlen_func", "");
    c10::Stack stack;
    stack.emplace_back(q);
    stack.emplace_back(k);
    stack.emplace_back(v);
    stack.emplace_back(cu_seqlens_q);
    stack.emplace_back(cu_seqlens_k);
    stack.emplace_back(block_table.has_value() ? c10::IValue(*block_table)
                                               : c10::IValue());
    stack.emplace_back(static_cast<int64_t>(max_seqlen_q));
    stack.emplace_back(static_cast<int64_t>(max_seqlen_k));
    stack.emplace_back(static_cast<double>(softmax_scale));
    stack.emplace_back(causal);
    stack.emplace_back(static_cast<int64_t>(window_size_left));
    stack.emplace_back(static_cast<int64_t>(window_size_right));
    stack.emplace_back(static_cast<double>(softcap));
    op.callBoxed(&stack);

    TORCH_CHECK(stack.size() == 1,
                "Mate flash_attn_varlen_func returned an invalid result");
    return stack.front().toTensorVector();
  }
};

}  // namespace infini::ops::linked::torch::moore

namespace infini::ops {

void Operator<FlashAttnVarlenFunc, Device::Type::kMoore, 16>::operator()(
    const Tensor q, const Tensor k, const Tensor v, const Tensor cu_seqlens_q,
    const Tensor cu_seqlens_k, const std::optional<Tensor> alibi_slopes,
    const std::optional<Tensor> block_table, const int64_t max_seqlen_q,
    const int64_t max_seqlen_k, const double dropout_p,
    const std::optional<double> softmax_scale, const bool causal,
    const std::vector<int64_t> window_size, const double softcap,
    const bool deterministic, const bool return_attn_probs, Tensor out,
    std::optional<Tensor> softmax_lse, std::optional<Tensor> s_dmask) const {
  using Delegate = linked::torch::TorchFlashAttnVarlenFunc<
      linked::torch::moore::MateFlashAttnVarlen>;
  if (!delegate_) {
    delegate_ = std::make_unique<Delegate>(
        q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
        max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
        window_size, softcap, deterministic, return_attn_probs, out,
        softmax_lse, s_dmask);
  }
  delegate_->set_stream(stream_);
  (*delegate_)(q, k, v, cu_seqlens_q, cu_seqlens_k, alibi_slopes, block_table,
               max_seqlen_q, max_seqlen_k, dropout_p, softmax_scale, causal,
               window_size, softcap, deterministic, return_attn_probs, out,
               softmax_lse, s_dmask);
}

}  // namespace infini::ops
