#ifndef INFINI_OPS_KUNLUN_SOFTMAX_KERNEL_H_
#define INFINI_OPS_KUNLUN_SOFTMAX_KERNEL_H_

#include <stdexcept>

#include "base/softmax.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun softmax over the last dimension. Rows at stride dim for both in/out
// (any leading-dim layout is folded into the row count when inner strides
// match the contiguous pattern; otherwise rejected).
template <>
class Operator<Softmax, Device::Type::kKunlun> : public Softmax {
 public:
  using Softmax::Softmax;

  bool capture_safe() const override { return true; }

  static bool supports(const Tensor input, const int64_t dim,
                       const std::optional<DataType> dtype, const Tensor out) {
    (void)dtype;
    (void)out;
    if (!input.IsContiguous() || !out.IsContiguous()) return false;
    // Row-walk kernel reduces the last dimension only.
    return input.ndim() == 1 || dim == -1 || dim == (int64_t)input.ndim() - 1;
  }

  void operator()(const Tensor input, const int64_t dim,
                  const std::optional<DataType> dtype,
                  Tensor out) const override {
    (void)dim;
    (void)dtype;
    if (row_count_ == 0 || dim_size_ == 0) {
      return;
    }
    if (dtype.has_value() && *dtype != out_type_) {
      throw std::runtime_error("kunlun softmax: dtype conversion unsupported");
    }
    // Inner-row layout: stride over the reduced dim must be 1 for the
    // row-walk kernel; the row stride comes from input_strides_[-1]...
    // input_strides_[-1] is the reduced dim's stride. Row-to-row stride is
    // input_strides_[-2] when ndim >= 2.
    if (input_strides_.back() != 1 || out_strides_.back() != 1) {
      throw std::runtime_error(
          "kunlun softmax: reduced dim must have stride 1");
    }
    const Tensor::Stride in_row_stride =
        input_strides_.size() >= 2 ? input_strides_[input_strides_.size() - 2]
                                   : 0;
    const Tensor::Stride out_row_stride =
        out_strides_.size() >= 2 ? out_strides_[out_strides_.size() - 2] : 0;

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out_type_);
    kunlun_launch::LaunchSoftmax(
        tag, out.data(), input.data(), (unsigned int)row_count_, (int)dim_size_,
        (long long)in_row_stride, (long long)out_row_stride, stream);
  }
};

}  // namespace infini::ops

#endif
