#ifndef INFINI_OPS_KUNLUN_SUB_KERNEL_H_
#define INFINI_OPS_KUNLUN_SUB_KERNEL_H_

#include "base/sub.h"
#include "host_range_profiler.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/ops/relu/kernel.h"  // detail::KunlunUploadBinaryMeta
#include "native/kunlun/runtime_.h"

namespace infini::ops {

namespace detail {

// Broadcast `input` strides against `out`'s shape (right-aligned broadcast).
inline Tensor::Strides KunlunBroadcastStrides(const Tensor& input,
                                              const Tensor& out) {
  const Tensor::Shape out_shape{out.shape()};
  const Tensor::Strides input_strides{input.strides()};
  Tensor::Strides result(out_shape.size(), 0);
  const Tensor::Size offset = out_shape.size() - input.ndim();
  for (Tensor::Size i = 0; i < input.ndim(); ++i) {
    result[offset + i] = input.size(i) == 1 ? 0 : input_strides[i];
  }
  return result;
}

}  // namespace detail

template <>
class Operator<Sub, Device::Type::kKunlun> : public Sub {
 public:
  using Sub::Sub;

  ~Operator() {
    if (d_metadata_ != nullptr) {
      Runtime<Device::Type::kKunlun>::Free(d_metadata_);
    }
  }

  void operator()(const Tensor input, const Tensor other, const double alpha,
                  Tensor out) const override {
    if (input.numel() == 0) {
      return;
    }
    [[maybe_unused]] HostRangeScope host_range_backend_submit{
        HostRangeLayer::kBackendSubmit};
    if (alpha != 1.0) {
      throw std::runtime_error("kunlun sub: only alpha == 1.0 is supported");
    }

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out_type_);

    // Upload shape/stride metadata for the (possibly broadcast/strided)
    // operands. Broadcasted strides are relative to the output layout.
    const Tensor::Shape out_shape{out.shape()};
    // Right-align-pad lower-rank operands: metadata buffer is sized by
    // out.ndim() and broadcast strides are relative to the output layout.
    const Tensor::Shape in_shape =
        detail::KunlunPadShape(input.shape(), out.ndim());
    const Tensor::Shape other_shape =
        detail::KunlunPadShape(other.shape(), out.ndim());
    const Tensor::Strides out_strides{out.strides()};
    const Tensor::Strides in_strides =
        detail::KunlunBroadcastStrides(input, out);
    const Tensor::Strides other_strides =
        detail::KunlunBroadcastStrides(other, out);
    d_metadata_ = detail::KunlunUploadBinaryMeta(
        out_shape, in_shape, other_shape, out_strides, in_strides,
        other_strides, d_out_shape_, d_in_shape_, d_other_shape_,
        d_out_strides_, d_in_strides_, d_other_strides_);

    kunlun_launch::LaunchSub(
        tag, out.data(), input.data(), other.data(),
        reinterpret_cast<const unsigned long long*>(d_out_shape_),
        reinterpret_cast<const unsigned long long*>(d_in_shape_),
        reinterpret_cast<const unsigned long long*>(d_other_shape_),
        reinterpret_cast<const long long*>(d_out_strides_),
        reinterpret_cast<const long long*>(d_in_strides_),
        reinterpret_cast<const long long*>(d_other_strides_), out.numel(),
        out.ndim(), out.IsContiguous(),
        input.shape() == out.shape() && input.IsContiguous(),
        other.shape() == out.shape() && other.IsContiguous(), stream);
  }

 private:
  // Mutable: metadata upload happens on first invocation; the buffer is
  // freed in the destructor.
  mutable std::byte* d_metadata_{nullptr};
  mutable Tensor::Size* d_out_shape_{nullptr};
  mutable Tensor::Size* d_in_shape_{nullptr};
  mutable Tensor::Size* d_other_shape_{nullptr};
  mutable Tensor::Stride* d_out_strides_{nullptr};
  mutable Tensor::Stride* d_in_strides_{nullptr};
  mutable Tensor::Stride* d_other_strides_{nullptr};
};

}  // namespace infini::ops

#endif
