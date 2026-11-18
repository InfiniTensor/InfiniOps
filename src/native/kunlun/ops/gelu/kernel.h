#ifndef INFINI_OPS_KUNLUN_GELU_KERNEL_H_
#define INFINI_OPS_KUNLUN_GELU_KERNEL_H_

#include <stdexcept>

#include "base/gelu.h"
#include "host_range_profiler.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/ops/add/kernel.h"  // detail::KunlunDtypeTag
#include "native/kunlun/runtime_.h"

namespace infini::ops {

template <>
class Operator<Gelu, Device::Type::kKunlun> : public Gelu {
 public:
  Operator(const Tensor input, const std::string approximate, Tensor out)
      : Gelu{input, approximate, out} {
    if (approximate_ != "none" && approximate_ != "tanh") {
      throw std::runtime_error(
          "kunlun gelu: only approximate in {none, tanh} is supported");
    }
    d_metadata_ = detail::KunlunUploadUnaryMeta(
        input_shape_, out_shape_, input_strides_, out_strides_, d_input_shape_,
        d_out_shape_, d_input_strides_, d_out_strides_);
  }

  Operator(const Tensor input, Tensor out) : Operator{input, "none", out} {}

  ~Operator() {
    if (d_metadata_ != nullptr) {
      Runtime<Device::Type::kKunlun>::Free(d_metadata_);
    }
  }

  void operator()(const Tensor input, const std::string approximate,
                  Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }
    [[maybe_unused]] HostRangeScope host_range_backend_submit{
        HostRangeLayer::kBackendSubmit};
    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    kunlun_launch::LaunchGelu(
        detail::KunlunDtypeTag(out_type_), out.data(), input.data(),
        reinterpret_cast<const unsigned long long*>(d_input_shape_),
        reinterpret_cast<const unsigned long long*>(d_out_shape_),
        reinterpret_cast<const long long*>(d_input_strides_),
        reinterpret_cast<const long long*>(d_out_strides_), output_size_, ndim_,
        is_out_contiguous_, is_input_contiguous_,
        approximate_ == "tanh" ? 1 : 0, stream);
  }

 private:
  std::byte* d_metadata_{nullptr};
  Tensor::Size* d_input_shape_{nullptr};
  Tensor::Size* d_out_shape_{nullptr};
  Tensor::Stride* d_input_strides_{nullptr};
  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
