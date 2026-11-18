#ifndef INFINI_OPS_KUNLUN_COPY_KERNEL_H_
#define INFINI_OPS_KUNLUN_COPY_KERNEL_H_

#include <cstring>
#include <stdexcept>

#include "base/copy.h"
#include "native/kunlun/launch2_declarations.h"
#include "native/kunlun/runtime_.h"

namespace infini::ops {

// Kunlun copy: broadcast-aware element copy, same dtype. A same-view
// in-place call is a no-op; contiguous pairs use a flat fast path.
template <>
class Operator<Copy, Device::Type::kKunlun> : public Copy {
 public:
  using Copy::Copy;

  bool capture_safe() const override { return true; }

  void operator()(const Tensor src, const bool non_blocking,
                  Tensor out) const override {
    (void)non_blocking;
    if (output_size_ == 0) {
      return;
    }
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);

    // The flat path copies `output_size_` linear elements from `src`; it is
    // only valid when the input has exactly as many elements (no broadcast).
    const bool flat = is_input_contiguous_ && is_out_contiguous_ &&
                      input_type_ == out_type_ && src.numel() == output_size_;
    if (flat && SameView(src, out)) {
      return;
    }

    const int element_size = out.element_size();
    if (element_size != 2 && element_size != 4) {
      throw std::runtime_error("kunlun copy: unsupported element size");
    }

    if (flat) {
      // Plain linear copy via the strided kernel with rank-1 metadata.
      const long long shape1[1] = {(long long)output_size_};
      const long long one[1] = {1};
      kunlun_launch::LaunchCopy(element_size, out.data(), src.data(), 1, shape1,
                                one, one, (unsigned long long)output_size_,
                                stream);
      return;
    }

    if (out.ndim() > 8) {
      throw std::runtime_error("kunlun copy: rank exceeds 8");
    }
    long long shape[8];
    long long in_strides[8];
    long long out_strides[8];
    for (std::size_t i = 0; i < out.ndim(); ++i) {
      shape[i] = (long long)out_shape_[i];
      in_strides[i] = (long long)input_strides_[i];
      out_strides[i] = (long long)out_strides_[i];
    }
    kunlun_launch::LaunchCopy(element_size, out.data(), src.data(),
                              (int)out.ndim(), shape, in_strides, out_strides,
                              (unsigned long long)output_size_, stream);
  }
};

}  // namespace infini::ops

#endif
