#ifndef INFINI_OPS_METAX_MUL_SCALAR_H_
#define INFINI_OPS_METAX_MUL_SCALAR_H_

#include <algorithm>
#include <cstring>
#include <stdexcept>

#include "native/cuda/caster.cuh"
#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"
#include "native/cuda/ops/mul/kernel.cuh"

namespace infini::ops::metax_mul_scalar_detail {

// Shape/strides ride in a fixed-size struct passed by value at launch,
// matching the NVIDIA scalar Mul and the MetaX Matmul-style layouts
// (per_tensor_i8 uses the same pattern): no device metadata buffer, no
// packing/unpacking around the launch.
struct Layout {
  std::int64_t ndim = 0;
  std::int64_t shape[16]{};
  std::int64_t input_strides[16]{};
  std::int64_t out_strides[16]{};
  bool contiguous = false;
};

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
__global__ void MulScalarStridedKernel(T* __restrict__ out,
                                       const T* __restrict__ input, T alpha,
                                       std::size_t count, Layout layout) {
  const auto start =
      static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  const auto step = static_cast<std::size_t>(gridDim.x) * blockDim.x;
  for (std::size_t linear = start; linear < count; linear += step) {
    std::size_t remaining = linear;
    std::int64_t input_offset = 0, out_offset = 0;
    for (std::int64_t d = layout.ndim; d-- > 0;) {
      const auto coord = static_cast<std::int64_t>(
          remaining % static_cast<std::size_t>(layout.shape[d]));
      remaining /= static_cast<std::size_t>(layout.shape[d]);
      input_offset += coord * layout.input_strides[d];
      out_offset += coord * layout.out_strides[d];
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
    // The by-value Layout carries up to 16 dims (same bound as NVIDIA).
    if (ndim > 16 || out.shape() != input.shape() ||
        out.strides().size() != ndim || input.strides().size() != ndim)
      return false;
    // The strided kernel indexes linearly over the output; require a
    // contiguous output so the linear index maps directly.
    return out.IsContiguous();
  }

  MetaxScalarMul(const Tensor input, const double alpha, Tensor out)
      : input_meta_(input), out_meta_(out), alpha_(alpha) {
    count_ = out.numel();
    layout_.ndim = static_cast<std::int64_t>(input.ndim());
    layout_.contiguous = input.IsContiguous() && out.IsContiguous();
    for (std::size_t d = 0; d < input.ndim(); ++d) {
      layout_.shape[d] = static_cast<std::int64_t>(input.shape()[d]);
      layout_.input_strides[d] = static_cast<std::int64_t>(input.strides()[d]);
      layout_.out_strides[d] = static_cast<std::int64_t>(out.strides()[d]);
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
          if (layout_.contiguous) {
            metax_mul_scalar_detail::MulScalarContiguousKernel<T>
                <<<blocks, kThreads, 0, stream>>>(d_out, d_input, scalar,
                                                  count_);
          } else {
            metax_mul_scalar_detail::MulScalarStridedKernel<T>
                <<<blocks, kThreads, 0, stream>>>(d_out, d_input, scalar,
                                                  count_, layout_);
          }
        },
        "MetaxScalarMul::Run");
  }

 private:
  Tensor input_meta_;
  Tensor out_meta_;
  double alpha_;
  std::size_t count_{0};
  metax_mul_scalar_detail::Layout layout_{};
};

}  // namespace infini::ops

#endif
