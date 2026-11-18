#ifndef INFINI_OPS_KUNLUN_RMS_NORM_KERNEL_H_
#define INFINI_OPS_KUNLUN_RMS_NORM_KERNEL_H_

#include <stdexcept>

#include "base/rms_norm.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun RMSNorm. The generic operator() contract: rows of length
// input.size(-1); any leading dims; input rows at element stride
// input_strides_[-2]-based layout — flattened here as row_stride =
// input_strides_[input_strides_.size()-2] when ndim >= 2 else stride = 0
// (single row). Weight must be contiguous (stride 1), which InfiniLM's
// parameter storage guarantees.
template <>
class Operator<RmsNorm, Device::Type::kKunlun> : public RmsNorm {
 public:
  using RmsNorm::RmsNorm;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    if (batch_size_ == 0 || dim_ == 0) {
      return;
    }
    const Tensor::Stride row_stride =
        input_strides_.size() >= 2 ? input_strides_[input_strides_.size() - 2]
                                   : 0;

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out.dtype());

    // Rows = product of all leading dims = batch_size_ * nhead_ (per the
    // base-ctor convention for rank>=3) or batch_size_ for rank 2.
    unsigned long long rows;
    if (out_shape_.size() >= 3) {
      rows = 1;
      for (std::size_t i = 0; i + 1 < out_shape_.size(); ++i) {
        rows *= out_shape_[i];
      }
    } else {
      rows = out_shape_.size() == 2 ? out_shape_[0] : 1;
    }

    kunlun_launch::LaunchRmsNorm(tag, out.data(), input.data(), weight.data(),
                                 (unsigned int)rows, (int)dim_, eps,
                                 (long long)row_stride, stream);
  }
};

}  // namespace infini::ops

#endif
