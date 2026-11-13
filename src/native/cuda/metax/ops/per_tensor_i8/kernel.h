#ifndef INFINI_OPS_METAX_PER_TENSOR_I8_KERNEL_H_
#define INFINI_OPS_METAX_PER_TENSOR_I8_KERNEL_H_

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/metax/caster.cuh"
#include "native/cuda/metax/runtime_.h"

namespace infini::ops {
namespace metax_per_tensor_i8_detail {

struct Layout {
  std::int64_t ndim = 0, shape[4]{}, x_stride[4]{}, packed_stride[4]{};
};

__device__ std::int64_t Offset(std::size_t linear, const Layout& l,
                               bool packed) {
  std::int64_t offset = 0;
  for (std::int64_t d = l.ndim; d-- > 0;) {
    auto coord = linear % l.shape[d];
    linear /= l.shape[d];
    offset += coord * (packed ? l.packed_stride[d] : l.x_stride[d]);
  }
  return offset;
}

__device__ void AtomicMax(float* ptr, float value) {
  auto bits = reinterpret_cast<int*>(ptr);
  int old = *bits, assumed;
  do {
    assumed = old;
    old = atomicCAS(bits, assumed,
                    __float_as_int(fmaxf(value, __int_as_float(assumed))));
  } while (old != assumed);
}

template <typename T>
__global__ void Absmax(const T* x, float* scale, Layout l, std::size_t count) {
  float maximum = 0;
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x)
    maximum = fmaxf(maximum, fabsf(float(x[Offset(i, l, false)])));
  AtomicMax(scale, maximum / 127.0f);
}

template <typename T>
__global__ void Quant(const T* x, std::int8_t* packed, const float* scale,
                      Layout l, std::size_t count) {
  const float inverse = 1.0f / scale[0];
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x) {
    const float value = float(x[Offset(i, l, false)]) * inverse;
    const float rounded = floorf(fabsf(value) + 0.5f);
    int q = value >= 0 ? int(rounded) : -int(rounded);
    q = q > 127 ? 127 : (q < -127 ? -127 : q);
    packed[Offset(i, l, true)] = std::int8_t(q);
  }
}

template <typename T>
__global__ void Dequant(T* x, const std::int8_t* packed, const float* scale,
                        Layout l, std::size_t count) {
  for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < count;
       i += std::size_t(gridDim.x) * blockDim.x)
    x[Offset(i, l, false)] = T(float(packed[Offset(i, l, true)]) * scale[0]);
}

inline bool Supports(Tensor x, Tensor packed, Tensor scale,
                     const std::optional<Tensor> zero, bool quant,
                     bool is_static = true) {
  if (zero ||
      (x.dtype() != DataType::kFloat16 && x.dtype() != DataType::kFloat32 &&
       x.dtype() != DataType::kBFloat16) ||
      packed.dtype() != DataType::kInt8 ||
      scale.dtype() != DataType::kFloat32 || scale.numel() != 1 ||
      x.device() != packed.device() || x.device() != scale.device() ||
      x.shape() != packed.shape() || x.ndim() > 4 ||
      x.strides().size() != x.ndim() ||
      packed.strides().size() != packed.ndim() ||
      !(quant ? x.IsContiguous() : packed.IsContiguous()) ||
      (quant && !is_static && !scale.IsContiguous()))
    return false;
  for (std::size_t d = 0; d < x.ndim(); ++d)
    if (x.strides()[d] < 0 || packed.strides()[d] < 0 ||
        x.shape()[d] > static_cast<std::size_t>(INT64_MAX))
      return false;
  return true;
}

struct Metadata {
  DataType dtype;
  Layout layout{};
  std::size_t count;
  Device device;
  Metadata(Tensor x, Tensor packed)
      : dtype(x.dtype()), count(x.numel()), device(x.device()) {
    layout.ndim = x.ndim();
    for (std::size_t d = 0; d < x.ndim(); ++d) {
      layout.shape[d] = x.shape()[d];
      layout.x_stride[d] = x.strides()[d];
      layout.packed_stride[d] = packed.strides()[d];
    }
  }
  bool Matches(Tensor x, Tensor packed) const {
    if (x.dtype() != dtype || x.device() != device ||
        x.ndim() != static_cast<std::size_t>(layout.ndim))
      return false;
    for (std::size_t d = 0; d < x.ndim(); ++d)
      if (x.shape()[d] != static_cast<std::size_t>(layout.shape[d]) ||
          x.strides()[d] != layout.x_stride[d] ||
          packed.strides()[d] != layout.packed_stride[d])
        return false;
    return true;
  }
};

template <typename T>
void SubmitQuant(const void* x, std::int8_t* packed, float* scale, Layout l,
                 std::size_t count, bool is_static,
                 Runtime<Device::Type::kMetax>::Stream stream) {
  const unsigned blocks =
      unsigned(std::min<std::size_t>((count - 1) / 256 + 1, 65535));
  if (!is_static) {
    Absmax<T>
        <<<blocks, 256, 0, stream>>>(static_cast<const T*>(x), scale, l, count);
  }
  Quant<T><<<blocks, 256, 0, stream>>>(static_cast<const T*>(x), packed, scale,
                                       l, count);
}

template <typename T>
void Dequantize(void* x, const std::int8_t* packed, const float* scale,
                Layout l, std::size_t count,
                Runtime<Device::Type::kMetax>::Stream stream) {
  const unsigned blocks =
      unsigned(std::min<std::size_t>((count - 1) / 256 + 1, 65535));
  Dequant<T>
      <<<blocks, 256, 0, stream>>>(static_cast<T*>(x), packed, scale, l, count);
}

inline Tensor Checked(Tensor x, Tensor packed, Tensor scale,
                      const std::optional<Tensor> zero, bool quant,
                      bool is_static = true) {
  if (!Supports(x, packed, scale, zero, quant, is_static))
    throw std::invalid_argument("unsupported per-tensor INT8 operands");
  return x;
}

}  // namespace metax_per_tensor_i8_detail

template <>
class Operator<PerTensorQuantI8, Device::Type::kMetax, 0>
    : public PerTensorQuantI8 {
 public:
  static bool supports(const Tensor x, Tensor packed, Tensor scale,
                       const std::optional<Tensor> zero, const bool is_static) {
    return metax_per_tensor_i8_detail::Supports(x, packed, scale, zero, true,
                                                is_static);
  }
  Operator(const Tensor x, Tensor packed, Tensor scale,
           const std::optional<Tensor> zero, const bool is_static)
      : PerTensorQuantI8(x, packed, scale, zero, is_static),
        meta_(metax_per_tensor_i8_detail::Checked(x, packed, scale, zero, true,
                                                  is_static),
              packed),
        is_static_(is_static) {}
  bool capture_safe() const override { return true; }
  void operator()(const Tensor x, Tensor packed, Tensor scale,
                  const std::optional<Tensor> zero,
                  const bool is_static) const override {
    if (!supports(x, packed, scale, zero, is_static) ||
        !meta_.Matches(x, packed) || is_static != is_static_)
      throw std::invalid_argument("unsupported per-tensor INT8 operands");
    auto stream = static_cast<Runtime<Device::Type::kMetax>::Stream>(
        stream_ ? stream_ : 0);
    DispatchFunc<
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        {static_cast<int64_t>(meta_.dtype)},
        [&](auto type_tag) {
          using T = TypeMapType<Device::Type::kMetax, ListGet<0>(type_tag)>;
          metax_per_tensor_i8_detail::template SubmitQuant<T>(
              x.data(), static_cast<std::int8_t*>(packed.data()),
              static_cast<float*>(scale.data()), meta_.layout, meta_.count,
              is_static, stream);
        },
        "MetaxPerTensorQuantI8");
  }

 private:
  metax_per_tensor_i8_detail::Metadata meta_;
  bool is_static_;
};

template <>
class Operator<PerTensorDequantI8, Device::Type::kMetax, 0>
    : public PerTensorDequantI8 {
 public:
  static bool supports(Tensor x, const Tensor packed, const Tensor scale,
                       const std::optional<Tensor> zero) {
    return metax_per_tensor_i8_detail::Supports(x, packed, scale, zero, false);
  }
  Operator(Tensor x, const Tensor packed, const Tensor scale,
           const std::optional<Tensor> zero)
      : PerTensorDequantI8(x, packed, scale, zero),
        meta_(
            metax_per_tensor_i8_detail::Checked(x, packed, scale, zero, false),
            packed) {}
  bool capture_safe() const override { return true; }
  void operator()(Tensor x, const Tensor packed, const Tensor scale,
                  const std::optional<Tensor> zero) const override {
    if (!supports(x, packed, scale, zero) || !meta_.Matches(x, packed))
      throw std::invalid_argument("unsupported per-tensor INT8 operands");
    auto stream = static_cast<Runtime<Device::Type::kMetax>::Stream>(
        stream_ ? stream_ : 0);
    DispatchFunc<
        List<DataType::kFloat16, DataType::kBFloat16, DataType::kFloat32>>(
        {static_cast<int64_t>(meta_.dtype)},
        [&](auto type_tag) {
          using T = TypeMapType<Device::Type::kMetax, ListGet<0>(type_tag)>;
          metax_per_tensor_i8_detail::template Dequantize<T>(
              x.data(), static_cast<const std::int8_t*>(packed.data()),
              static_cast<const float*>(scale.data()), meta_.layout,
              meta_.count, stream);
        },
        "MetaxPerTensorDequantI8");
  }

 private:
  metax_per_tensor_i8_detail::Metadata meta_;
};
}  // namespace infini::ops

#endif  // INFINI_OPS_METAX_PER_TENSOR_I8_KERNEL_H_
