#ifndef INFINI_OPS_TENSOR_H_
#define INFINI_OPS_TENSOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "data_type.h"
#include "device.h"
#include "hash.h"
#include "small_vector.h"

namespace infini::ops {

class Tensor {
 public:
  using Size = std::size_t;

  using Stride = std::ptrdiff_t;

  using Index = Stride;

  // Tensors used on the hot path (model decode, elementwise, GEMM) have
  // 2-4 dims. Storing shape/strides inline up to 6 elements skips the heap
  // allocation that `std::vector` would do on every `Tensor` copy in the
  // shim's `to_ops_tensor` / `Operator::Call` plumbing.
  using Shape = SmallVector<Size, 6>;

  using Strides = SmallVector<Stride, 6>;

  template <typename Shape>
  Tensor(void* data, const Shape& shape)
      : data_{data},
        shape_{shape},
        dtype_{DefaultDataType()},
        device_{DefaultDevice()},
        strides_{DefaultStrides(shape)} {}

  template <typename Shape>
  Tensor(void* data, const Shape& shape, const DataType& dtype)
      : data_{data},
        shape_{shape},
        dtype_{dtype},
        device_{DefaultDevice()},
        strides_{DefaultStrides(shape)} {}

  template <typename Shape>
  Tensor(void* data, const Shape& shape, const Device& device)
      : data_{data},
        shape_{shape},
        dtype_{DefaultDataType()},
        device_{device},
        strides_{DefaultStrides(shape)} {}

  template <typename Shape>
  Tensor(void* data, const Shape& shape, const DataType& dtype,
         const Device& device)
      : data_{data},
        shape_{shape},
        dtype_{dtype},
        device_{device},
        strides_{DefaultStrides(shape)} {}

  template <typename Shape, typename Strides>
  Tensor(void* data, const Shape& shape, const DataType& dtype,
         const Device& device, const Strides& strides)
      : data_{data},
        shape_{shape},
        dtype_{dtype},
        device_{device},
        strides_{strides} {}

  Tensor(void* data, std::initializer_list<Size> shape, const DataType& dtype,
         const Device& device, std::initializer_list<Stride> strides);

  Tensor operator[](const Index& index) const;

  void*& data();

  const void* data() const;

  const DataType& dtype() const;

  const Device& device() const;

  const Shape& shape() const;

  const Strides& strides() const;

  Size size(const Index& index) const;

  Stride stride(const Index& index) const;

  Size ndim() const;

  Size element_size() const;

  Size numel() const;

  Tensor T() const;

  std::string ToString() const;

  bool HasBroadcastDim() const;

  bool IsContiguous() const;

 private:
  static const DataType DefaultDataType();

  static Device DefaultDevice();

  static Strides DefaultStrides(const Shape& shape);

  std::string ToStringHelper() const;

  bool IsMergeable(Size dim_start, Size dim_end) const;

  void* data_{nullptr};

  Shape shape_;

  const DataType dtype_;

  Device device_;

  Strides strides_;
};

}  // namespace infini::ops

template <>
struct std::hash<infini::ops::Tensor> {
  std::size_t operator()(const infini::ops::Tensor& tensor) const {
    std::size_t seed{0};

    for (const auto& size : tensor.shape()) {
      HashCombine(seed, size);
    }

    HashCombine(seed, tensor.dtype());

    HashCombine(seed, tensor.device());

    for (const auto& stride : tensor.strides()) {
      HashCombine(seed, stride);
    }

    return seed;
  }
};

template <>
struct std::equal_to<infini::ops::Tensor> {
  bool operator()(const infini::ops::Tensor& a,
                  const infini::ops::Tensor& b) const {
    return a.dtype() == b.dtype() && a.device() == b.device() &&
           a.shape() == b.shape() && a.strides() == b.strides();
  }
};

#endif
