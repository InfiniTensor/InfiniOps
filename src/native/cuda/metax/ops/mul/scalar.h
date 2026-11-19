#ifndef INFINI_OPS_METAX_MUL_SCALAR_H_
#define INFINI_OPS_METAX_MUL_SCALAR_H_

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "native/cuda/caster.cuh"
#include "native/cuda/device_buffer.h"
#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/mul/kernel.cuh"

namespace infini::ops::metax_mul_scalar_detail {

template <typename T>
__global__ void MulScalarContiguousKernel(T* __restrict__ out,
                                          const T* __restrict__ input, T alpha,
                                          std::size_t count) {
  const auto start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = start; linear < count; linear += step) {
    out[linear] = MulOp<Device::Type::kMetax>{}(input[linear], alpha);
  }
}

template <typename T>
__global__ void MulScalarStridedKernel(
    T* __restrict__ out, const T* __restrict__ input, T alpha,
    std::size_t count, std::int64_t ndim, const std::size_t* __restrict__ shape,
    const std::ptrdiff_t* __restrict__ input_strides,
    const std::ptrdiff_t* __restrict__ out_strides) {
  const auto start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = start; linear < count; linear += step) {
    std::size_t remaining = linear;
    std::ptrdiff_t input_offset = 0, out_offset = 0;
    for (std::int64_t d = ndim; d-- > 0;) {
      const auto coord = static_cast<std::ptrdiff_t>(
          remaining % static_cast<std::size_t>(shape[d]));
      remaining /= static_cast<std::size_t>(shape[d]);
      input_offset += coord * input_strides[d];
      out_offset += coord * out_strides[d];
    }
    out[out_offset] = MulOp<Device::Type::kMetax>{}(input[input_offset], alpha);
  }
}

}  // namespace infini::ops::metax_mul_scalar_detail

namespace infini::ops {

// Scalar-multiplier provider for MetaX: `Mul(tensor, double, tensor)`.
// The shared `CudaMul` only implements the tensor x tensor ctor; the
// scalar ctor there is protected and its `operator()` rejects scalar
// mode, so dispatch of the scalar overload finds no MetaX provider.
// The DeepSeek-V2 MoE router scale (routed_scaling_factor) needs it.
class MetaxScalarMul {
 public:
  static bool supports(const Tensor input, const double, Tensor out) {
    if (input.device().type() != Device::Type::kMetax ||
        input.device() != out.device() || input.dtype() != out.dtype())
      return false;
    switch (input.dtype()) {
      case DataType::kFloat16:
      case DataType::kBFloat16:
      case DataType::kFloat32:
      case DataType::kFloat64:
        break;
      default:
        return false;
    }
    const auto ndim = input.ndim();
    if (out.shape() != input.shape() || out.strides().size() != ndim ||
        input.strides().size() != ndim)
      return false;
    // The strided kernel indexes linearly over the output; require a
    // contiguous output so the linear index maps directly.
    return out.IsContiguous();
  }

  MetaxScalarMul(const Tensor input, const double alpha, Tensor out)
      : input_meta_(input), out_meta_(out), alpha_(alpha) {
    count_ = out.numel();
    contiguous_ = input.IsContiguous() && out.IsContiguous();
    if (contiguous_) {
      return;
    }
    shape_.assign(input.shape().begin(), input.shape().end());
    input_strides_.assign(input.strides().begin(), input.strides().end());
    out_strides_.assign(out.strides().begin(), out.strides().end());
    // The strided kernel reads this metadata asynchronously, so the
    // allocation must outlive the launch: keep it as a member instead
    // of a Run-scoped buffer whose Free would race the kernel.
    const std::size_t shape_bytes = shape_.size() * sizeof(Tensor::Size);
    strides_bytes_ = shape_.size() * sizeof(Tensor::Stride);
    const std::size_t metadata_bytes = shape_bytes + 2 * strides_bytes_;
    std::vector<std::byte> host(metadata_bytes);
    std::memcpy(host.data(), shape_.data(), shape_bytes);
    std::memcpy(host.data() + shape_bytes, input_strides_.data(),
                strides_bytes_);
    std::memcpy(host.data() + shape_bytes + strides_bytes_, out_strides_.data(),
                strides_bytes_);
    metadata_ptr_ = metadata_.Allocate(metadata_bytes);
    if (Runtime<Device::Type::kMetax>::Memcpy(
            const_cast<std::byte*>(metadata_ptr_), host.data(), metadata_bytes,
            Runtime<Device::Type::kMetax>::kMemcpyHostToDevice) !=
        Runtime<Device::Type::kMetax>::kSuccess) {
      throw std::runtime_error("MetaX scalar Mul metadata copy failed");
    }
  }

  void Run(const Tensor input, const double alpha, Tensor out,
           typename Runtime<Device::Type::kMetax>::Stream stream) const {
    if (!std::equal_to<Tensor>{}(input, input_meta_) ||
        !std::equal_to<Tensor>{}(out, out_meta_) || alpha != alpha_)
      throw std::invalid_argument(
          "MetaX scalar Mul metadata or alpha changed after Make");
    if (!count_) return;
    DispatchFunc<AllFloatTypes>(
        {static_cast<int64_t>(input.dtype())},
        [&](auto list_tag) {
          using T = TypeMapType<Device::Type::kMetax, ListGet<0>(list_tag)>;
          const auto scalar = Caster<Device::Type::kMetax>::Cast<T>(alpha_);
          auto* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          constexpr unsigned kThreads = 256;
          const auto blocks = static_cast<unsigned>(
              std::min<std::size_t>((count_ - 1) / kThreads + 1, 65535));
          if (contiguous_) {
            metax_mul_scalar_detail::MulScalarContiguousKernel<T>
                <<<blocks, kThreads, 0, stream>>>(d_out, d_input, scalar,
                                                  count_);
          } else {
            const std::size_t shape_bytes =
                shape_.size() * sizeof(Tensor::Size);
            metax_mul_scalar_detail::MulScalarStridedKernel<T>
                <<<blocks, kThreads, 0, stream>>>(
                    d_out, d_input, scalar, count_,
                    static_cast<std::int64_t>(shape_.size()),
                    reinterpret_cast<const Tensor::Size*>(metadata_ptr_),
                    reinterpret_cast<const Tensor::Stride*>(metadata_ptr_ +
                                                            shape_bytes),
                    reinterpret_cast<const Tensor::Stride*>(
                        metadata_ptr_ + shape_bytes + strides_bytes_));
          }
        },
        "MetaxScalarMul::Run");
  }

 private:
  Tensor input_meta_;
  Tensor out_meta_;
  double alpha_;
  std::size_t count_{0};
  bool contiguous_{false};
  std::size_t strides_bytes_{0};
  std::vector<Tensor::Size> shape_{};
  std::vector<Tensor::Stride> input_strides_{};
  std::vector<Tensor::Stride> out_strides_{};
  CudaDeviceBuffer<Runtime<Device::Type::kMetax>> metadata_;
  std::byte* metadata_ptr_{nullptr};
};

}  // namespace infini::ops

#endif
