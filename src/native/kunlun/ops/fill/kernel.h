#ifndef INFINI_OPS_KUNLUN_FILL_KERNEL_H_
#define INFINI_OPS_KUNLUN_FILL_KERNEL_H_

#include <stdexcept>

#include "base/fill.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun fill: broadcast-aware scalar fill (f32/f16/bf16/f64). The
// tensor-value overload is not supported (InfiniLM only uses the scalar).
template <>
class Operator<Fill, Device::Type::kKunlun> : public Fill {
 public:
  using Fill::Fill;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const double value,
                  Tensor out) const override {
    (void)input;
    if (out.numel() == 0) {
      return;
    }
    if (out.ndim() > 8) {
      throw std::runtime_error("kunlun fill: rank exceeds 8");
    }
    if (!valid_output(out, false)) {
      throw std::runtime_error("kunlun fill: invalid output layout");
    }
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    long long shape[8];
    long long out_strides[8];
    for (std::size_t i = 0; i < out.ndim(); ++i) {
      shape[i] = (long long)out_shape_[i];
      out_strides[i] = (long long)out_strides_[i];
    }
    if (out_type_ == DataType::kFloat32) {
      const float v = static_cast<float>(value);
      kunlun_launch::LaunchFill(4, 0, out.data(), (int)out.ndim(), shape,
                                out_strides, (unsigned long long)out.numel(), v,
                                stream);
    } else if (out_type_ == DataType::kFloat64) {
      throw std::runtime_error("kunlun fill: f64 unsupported");
    } else {
      const int tag = detail::KunlunDtypeTag(out_type_);
      const float v = static_cast<float>(value);
      kunlun_launch::LaunchFill(2, tag, out.data(), (int)out.ndim(), shape,
                                out_strides, (unsigned long long)out.numel(), v,
                                stream);
    }
  }

  void operator()(const Tensor input, const Tensor value,
                  Tensor out) const override {
    (void)input;
    (void)value;
    (void)out;
    throw std::runtime_error("kunlun fill: tensor value unsupported");
  }
};

}  // namespace infini::ops

#endif
