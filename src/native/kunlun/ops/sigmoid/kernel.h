#ifndef INFINI_OPS_KUNLUN_SIGMOID_KERNEL_H_
#define INFINI_OPS_KUNLUN_SIGMOID_KERNEL_H_

#include <stdexcept>

#include "base/sigmoid.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun sigmoid: flat, contiguous, f32/f16/bf16.
template <>
class Operator<Sigmoid, Device::Type::kKunlun> : public Sigmoid {
 public:
  using Sigmoid::Sigmoid;

  bool capture_safe() const override { return true; }

  static bool supports(const Tensor input, Tensor out) {
    return input.IsContiguous() && out.IsContiguous();
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }
    if (!is_input_contiguous_ || !is_out_contiguous_) {
      throw std::runtime_error("kunlun sigmoid: requires contiguous tensors");
    }
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out_type_);
    kunlun_launch::LaunchSigmoid(tag, out.data(), input.data(),
                                 (unsigned long long)output_size_, stream);
  }
};

}  // namespace infini::ops

#endif
