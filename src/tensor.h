#ifndef INFINI_OPS_TENSOR_H_
#define INFINI_OPS_TENSOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "data_type.h"
#include "device.h"

namespace infini::ops {

class Tensor {
 public:
  using Size = std::uint64_t;

  using Stride = std::int64_t;

  using Index = Stride;

  using Shape = std::vector<Size>;

  using Strides = std::vector<Stride>;

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

  const void* const& data() const;

  const DataType& dtype() const;

  const Device& device() const;

  const Shape& shape() const;

  const Strides& strides() const;

  Size size(const Index& index) const;

  Stride stride(const Index& index) const;

  Size ndim() const;

  Size element_size() const;

  Tensor T() const;

  std::string ToString() const;

 private:
  static const DataType& DefaultDataType();

  static Device DefaultDevice();

  static Strides DefaultStrides(const Shape& shape);

  std::string ToStringHelper() const;

  void* data_{nullptr};

  Shape shape_;

  const DataType& dtype_;

  Device device_;

  Strides strides_;
};

}  // namespace infini::ops

#endif
