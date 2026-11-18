#ifndef INFINI_OPS_KUNLUN_CAST_KERNEL_H_
#define INFINI_OPS_KUNLUN_CAST_KERNEL_H_

#include <stdexcept>

#include "base/cast.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun cast between f32/f16/bf16 (any direction). Flat, contiguous.
template <>
class Operator<Cast, Device::Type::kKunlun> : public Cast {
 public:
  using Cast::Cast;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }
    if (!is_input_contiguous_ || !is_out_contiguous_) {
      throw std::runtime_error("kunlun cast: requires contiguous tensors");
    }
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    if (input_dtype_ != DataType::kFloat32 &&
        input_dtype_ != DataType::kFloat16 &&
        input_dtype_ != DataType::kBFloat16) {
      throw std::runtime_error("kunlun cast: unsupported input dtype " +
                               std::to_string(static_cast<int>(input_dtype_)));
    }
    if (out_dtype_ != DataType::kFloat32 && out_dtype_ != DataType::kFloat16 &&
        out_dtype_ != DataType::kBFloat16) {
      throw std::runtime_error("kunlun cast: unsupported output dtype " +
                               std::to_string(static_cast<int>(out_dtype_)));
    }
    const int tag_in = detail::KunlunDtypeTag(input_dtype_);
    const int tag_out = detail::KunlunDtypeTag(out_dtype_);
    kunlun_launch::LaunchCast(tag_in, tag_out, out.data(), input.data(),
                              (unsigned long long)output_size_, stream);
  }
};

}  // namespace infini::ops

#endif
