#ifndef INFINI_OPS_KUNLUN_SILU_AND_MUL_KERNEL_H_
#define INFINI_OPS_KUNLUN_SILU_AND_MUL_KERNEL_H_

#include "base/silu_and_mul.h"
#include "host_range_profiler.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/ops/add/kernel.h"  // detail::KunlunDtypeTag
#include "native/kunlun/runtime_.h"

namespace infini::ops {

template <>
class Operator<SiluAndMul, Device::Type::kKunlun> : public SiluAndMul {
 public:
  Operator(const Tensor input, Tensor out) : SiluAndMul{input, out} {
    // Upload full shape/stride metadata for the Nd fallback path.
    using Backend = Runtime<Device::Type::kKunlun>;
    const std::size_t nd = ndim_;
    const std::size_t shape_bytes = nd * sizeof(Tensor::Size);
    const std::size_t stride_bytes = nd * sizeof(Tensor::Stride);
    const std::size_t meta_bytes = shape_bytes + 2 * stride_bytes;
    if (meta_bytes == 0) {
      return;
    }
    if (Backend::Malloc(reinterpret_cast<void**>(&d_metadata_), meta_bytes) !=
        Backend::kSuccess) {
      throw std::runtime_error("kunlun: silu_and_mul meta malloc failed");
    }
    std::vector<std::byte> host(meta_bytes);
    std::size_t offset = 0;
    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(host.data() + offset, out_shape_.data(), shape_bytes);
    offset += shape_bytes;
    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(host.data() + offset, out_strides_.data(), stride_bytes);
    offset += stride_bytes;
    d_in_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(host.data() + offset, input_strides_.data(), stride_bytes);
    if (Backend::Memcpy(d_metadata_, host.data(), meta_bytes,
                        Backend::kMemcpyHostToDevice) != Backend::kSuccess) {
      Backend::Free(d_metadata_);
      d_metadata_ = nullptr;
      throw std::runtime_error("kunlun: silu_and_mul meta copy failed");
    }
  }

  ~Operator() {
    if (d_metadata_ != nullptr) {
      Runtime<Device::Type::kKunlun>::Free(d_metadata_);
    }
  }

  void operator()(const Tensor input, Tensor out) const override {
    if (output_size_ == 0) {
      return;
    }
    [[maybe_unused]] HostRangeScope host_range_backend_submit{
        HostRangeLayer::kBackendSubmit};

    auto* stream = static_cast<XPUStream>(stream_ ? stream_ : nullptr);
    const int tag = detail::KunlunDtypeTag(out_type_);
    // Fast row-stride path only valid when there is a single row dim
    // (ndim==2): with more outer dims the uniform input_strides_[0] step
    // ignores middle-dim strides. Everything else goes to the Nd kernel.
    const bool row_stride_ok =
        is_input_contiguous_ || (ndim_ <= 2 && input_strides_.back() == 1);
    if (row_stride_ok && (is_out_contiguous_ || out_strides_.back() == 1)) {
      kunlun_launch::LaunchSiluAndMul(
          tag, out.data(), input.data(), output_size_ / hidden_size_,
          hidden_size_,
          is_input_contiguous_ ? 0 : static_cast<long long>(input_strides_[0]),
          is_out_contiguous_ ? 0 : static_cast<long long>(out_strides_[0]),
          is_input_contiguous_, is_out_contiguous_, stream);
    } else {
      kunlun_launch::LaunchSiluAndMulNd(
          tag, out.data(), input.data(),
          reinterpret_cast<const unsigned long long*>(d_out_shape_),
          reinterpret_cast<const long long*>(d_out_strides_),
          reinterpret_cast<const long long*>(d_in_strides_), ndim_,
          hidden_size_, stream);
    }
  }

 private:
  std::byte* d_metadata_{nullptr};
  Tensor::Size* d_out_shape_{nullptr};
  Tensor::Stride* d_out_strides_{nullptr};
  Tensor::Stride* d_in_strides_{nullptr};
};

}  // namespace infini::ops

#endif
