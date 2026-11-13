#ifndef INFINI_OPS_LINKED_TORCH_METAX_OPS_PAD_ATEN_H_
#define INFINI_OPS_LINKED_TORCH_METAX_OPS_PAD_ATEN_H_

#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>

#include <cassert>
#include <optional>
#include <vector>

#include "linked/torch/ops/pad.h"
#include "torch/metax/c10.h"

namespace infini::ops::linked::torch::metax {

struct AtenPad : C10<Device::Type::kMetax> {
  static void Call(at::Tensor out, at::Tensor input, std::vector<int64_t> pad,
                   double value) {
    static const auto op = c10::Dispatcher::singleton().findSchemaOrThrow(
        "aten::constant_pad_nd", "");
    c10::Stack stack;
    stack.reserve(4);
    stack.emplace_back(std::move(input));
    stack.emplace_back(c10::List<int64_t>(pad));
    stack.emplace_back(value);
    op.callBoxed(&stack);
    TORCH_CHECK(stack.size() == 1,
                "`constant_pad_nd` returned unexpected values.");
    out.copy_(stack.front().toTensor());
  }
};

}  // namespace infini::ops::linked::torch::metax

namespace infini::ops::linked::torch {

extern template class TorchPad< ::infini::ops::linked::torch::metax::AtenPad>;

}  // namespace infini::ops::linked::torch

namespace infini::ops {

template <>
class Operator<Pad, Device::Type::kMetax, 0>
    : public linked::torch::TorchPad<linked::torch::metax::AtenPad> {
 public:
  using linked::torch::TorchPad<linked::torch::metax::AtenPad>::TorchPad;
  using linked::torch::TorchPad<linked::torch::metax::AtenPad>::operator();
};

}  // namespace infini::ops

#endif  // INFINI_OPS_LINKED_TORCH_METAX_OPS_PAD_ATEN_H_
