#ifndef INFINI_OPS_TENSOR_H_
#define INFINI_OPS_TENSOR_H_

#include <cstdint>
#include <string>
#include <vector>

#include "data_type.h"

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
        strides_{DefaultStrides(shape)} {}

  template <typename Shape, typename Strides>
  Tensor(void* data, const Shape& shape, const DataType& dtype)
      : data_{data},
        shape_{shape},
        dtype_{dtype},
        strides_{DefaultStrides(shape)} {}

  template <typename Shape, typename Strides>
  Tensor(void* data, const Shape& shape, const DataType& dtype,
         const Strides& strides)
      : data_{data}, shape_{shape}, dtype_{dtype}, strides_{strides} {}

  Tensor(void* data, std::initializer_list<Size> shape, const DataType& dtype,
         std::initializer_list<Stride> strides);

  Tensor operator[](const Index& index) const;

  void*& data();

  const void* const& data() const;

  const DataType& dtype() const;

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

  static Strides DefaultStrides(const Shape& shape);

  std::string ToStringHelper() const;

  void* data_{nullptr};

  Shape shape_;

  const DataType& dtype_;

  Strides strides_;
};

}  // namespace infini::ops

#endif
