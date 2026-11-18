#ifndef INFINI_OPS_KUNLUN_RELU_KERNEL_H_
#define INFINI_OPS_KUNLUN_RELU_KERNEL_H_

#include <vector>

#include "base/relu.h"
#include "common/generic_utils.h"
#include "host_range_profiler.h"
#include "native/kunlun/elementwise_common.h"
#include "native/kunlun/runtime_.h"
#include "native/kunlun/runtime_utils.h"

namespace infini::ops {

namespace detail {
// Shared: upload shape/stride metadata for a unary elementwise op.
// Returns the device buffer; caller owns it (freed in the destructor).
inline std::byte* KunlunUploadUnaryMeta(
    const Tensor::Shape& input_shape, const Tensor::Shape& out_shape,
    const Tensor::Strides& input_strides, const Tensor::Strides& out_strides,
    Tensor::Size*& d_input_shape, Tensor::Size*& d_out_shape,
    Tensor::Stride*& d_input_strides, Tensor::Stride*& d_out_strides) {
  using Backend = Runtime<Device::Type::kKunlun>;
  const std::size_t ndim = out_shape.size();
  const std::size_t shape_bytes = ndim * sizeof(Tensor::Size);
  const std::size_t stride_bytes = ndim * sizeof(Tensor::Stride);
  const std::size_t meta_bytes = 2 * (shape_bytes + stride_bytes);
  if (meta_bytes == 0) {
    return nullptr;
  }

  std::vector<std::byte> host(meta_bytes);
  std::byte* d_meta = nullptr;
  if (Backend::Malloc(reinterpret_cast<void**>(&d_meta), meta_bytes) !=
      Backend::kSuccess) {
    throw std::runtime_error("kunlun: metadata malloc failed");
  }

  std::size_t offset = 0;
  d_input_shape = reinterpret_cast<Tensor::Size*>(d_meta + offset);
  std::memcpy(host.data() + offset, input_shape.data(), shape_bytes);
  offset += shape_bytes;
  d_out_shape = reinterpret_cast<Tensor::Size*>(d_meta + offset);
  std::memcpy(host.data() + offset, out_shape.data(), shape_bytes);
  offset += shape_bytes;
  d_input_strides = reinterpret_cast<Tensor::Stride*>(d_meta + offset);
  std::memcpy(host.data() + offset, input_strides.data(), stride_bytes);
  offset += stride_bytes;
  d_out_strides = reinterpret_cast<Tensor::Stride*>(d_meta + offset);
  std::memcpy(host.data() + offset, out_strides.data(), stride_bytes);

  if (Backend::Memcpy(d_meta, host.data(), meta_bytes,
                      Backend::kMemcpyHostToDevice) != Backend::kSuccess) {
    Backend::Free(d_meta);
    throw std::runtime_error("kunlun: metadata copy failed");
  }
  return d_meta;
}
}  // namespace detail

template <>
class Operator<Relu, Device::Type::kKunlun> : public Relu {
 public:
  Operator(const Tensor input, Tensor out) : Relu{input, out} {
    d_metadata_ = detail::KunlunUploadUnaryMeta(
        input_shape_, out_shape_, input_strides_, out_strides_, d_input_shape_,
        d_out_shape_, d_input_strides_, d_out_strides_);
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
    void* out_ptr = out.data();
    const void* in_ptr = input.data();
    bool out_c = is_out_contiguous_;
    bool in_c = is_input_contiguous_;
    // Overlapping in-place: make a contiguous copy of the input first.
    void* input_copy = nullptr;
    if (NeedsInputCopy(input, out)) {
      using Backend = Runtime<Device::Type::kKunlun>;
      if (Backend::Malloc(&input_copy, output_size_ * out.element_size()) !=
          Backend::kSuccess) {
        throw std::runtime_error("kunlun: relu input copy malloc failed");
      }
      // Snapshot the input before the kernel overwrites the shared storage.
      if (Backend::Memcpy(
              input_copy, input.data(), output_size_ * out.element_size(),
              Backend::kMemcpyDeviceToDevice) != Backend::kSuccess) {
        Backend::Free(input_copy);
        throw std::runtime_error("kunlun: relu input copy failed");
      }
      in_ptr = input_copy;
      in_c = true;
    }

    kunlun_launch::LaunchRelu(
        tag, out_ptr, in_ptr,
        reinterpret_cast<const unsigned long long*>(d_out_shape_),
        reinterpret_cast<const unsigned long long*>(d_input_shape_),
        reinterpret_cast<const long long*>(d_out_strides_),
        reinterpret_cast<const long long*>(d_input_strides_), output_size_,
        ndim_, out_c, in_c, stream);

    if (input_copy != nullptr) {
      Runtime<Device::Type::kKunlun>::Free(input_copy);
    }
  }

 private:
  std::byte* d_metadata_{nullptr};
  Tensor::Size* d_input_shape_{nullptr};
  Tensor::Size* d_out_shape_{nullptr};
  Tensor::Stride* d_input_strides_{nullptr};
  Tensor::Stride* d_out_strides_{nullptr};
};

namespace detail {

// Shared: upload shape/stride metadata for a binary elementwise op.
template <typename ShapeLike>
inline Tensor::Shape KunlunPadShape(const ShapeLike& shape, std::size_t ndim) {
  // Right-align-pad a lower-rank shape with leading 1s: the metadata buffer
  // is sized by the output's ndim, and broadcast strides are defined
  // relative to the output layout.
  Tensor::Shape result(ndim, 1);
  const std::size_t offset = ndim - shape.size();
  for (std::size_t i = 0; i < shape.size(); ++i) {
    result[offset + i] = shape[i];
  }
  return result;
}

inline std::byte* KunlunUploadBinaryMeta(
    const Tensor::Shape& out_shape, const Tensor::Shape& in_shape,
    const Tensor::Shape& other_shape, const Tensor::Strides& out_strides,
    const Tensor::Strides& in_strides, const Tensor::Strides& other_strides,
    Tensor::Size*& d_out_shape, Tensor::Size*& d_in_shape,
    Tensor::Size*& d_other_shape, Tensor::Stride*& d_out_strides,
    Tensor::Stride*& d_in_strides, Tensor::Stride*& d_other_strides) {
  using Backend = Runtime<Device::Type::kKunlun>;
  const std::size_t ndim = out_shape.size();
  const std::size_t shape_bytes = ndim * sizeof(Tensor::Size);
  const std::size_t stride_bytes = ndim * sizeof(Tensor::Stride);
  const std::size_t meta_bytes = 3 * (shape_bytes + stride_bytes);
  if (meta_bytes == 0) {
    return nullptr;
  }

  std::vector<std::byte> host(meta_bytes);
  std::byte* d_meta = nullptr;
  if (Backend::Malloc(reinterpret_cast<void**>(&d_meta), meta_bytes) !=
      Backend::kSuccess) {
    throw std::runtime_error("kunlun: binary metadata malloc failed");
  }

  std::size_t offset = 0;
  d_out_shape = reinterpret_cast<Tensor::Size*>(d_meta + offset);
  std::memcpy(host.data() + offset, out_shape.data(), shape_bytes);
  offset += shape_bytes;
  d_in_shape = reinterpret_cast<Tensor::Size*>(d_meta + offset);
  std::memcpy(host.data() + offset, in_shape.data(), shape_bytes);
  offset += shape_bytes;
  d_other_shape = reinterpret_cast<Tensor::Size*>(d_meta + offset);
  std::memcpy(host.data() + offset, other_shape.data(), shape_bytes);
  offset += shape_bytes;
  d_out_strides = reinterpret_cast<Tensor::Stride*>(d_meta + offset);
  std::memcpy(host.data() + offset, out_strides.data(), stride_bytes);
  offset += stride_bytes;
  d_in_strides = reinterpret_cast<Tensor::Stride*>(d_meta + offset);
  std::memcpy(host.data() + offset, in_strides.data(), stride_bytes);
  offset += stride_bytes;
  d_other_strides = reinterpret_cast<Tensor::Stride*>(d_meta + offset);
  std::memcpy(host.data() + offset, other_strides.data(), stride_bytes);

  if (Backend::Memcpy(d_meta, host.data(), meta_bytes,
                      Backend::kMemcpyHostToDevice) != Backend::kSuccess) {
    Backend::Free(d_meta);
    throw std::runtime_error("kunlun: binary metadata copy failed");
  }
  return d_meta;
}

}  // namespace detail

}  // namespace infini::ops

#endif
