#include "tensor.h"

#include <cassert>

namespace infini::ops {

Tensor::Tensor(void* data, std::initializer_list<Size> shape,
               const DataType& dtype, std::initializer_list<Stride> strides)
    : Tensor{data, decltype(shape_){shape}, dtype,
             decltype(strides_){strides}} {}

Tensor Tensor::operator[](const Index& index) const {
  return {reinterpret_cast<decltype(data_)>(
              reinterpret_cast<decltype(index)>(data_) +
              index * strides_[0] * element_size()),
          Shape{shape_.cbegin() + 1, shape_.cend()}, dtype_,
          Strides{strides_.cbegin() + 1, strides_.cend()}};
}

void*& Tensor::data() { return data_; }

const void* const& Tensor::data() const { return data_; }

const Tensor::Shape& Tensor::shape() const { return shape_; }

const DataType& Tensor::dtype() const { return dtype_; }

const Tensor::Strides& Tensor::strides() const { return strides_; }

Tensor::Size Tensor::size(const Index& index) const { return shape_[index]; }

Tensor::Stride Tensor::stride(const Index& index) const {
  return shape_[index];
}

Tensor::Size Tensor::ndim() const { return shape_.size(); }

Tensor::Size Tensor::element_size() const { return dtype_.element_size(); }

Tensor Tensor::T() const {
  return {data_, {shape_[1], shape_[0]}, dtype_, {strides_[1], strides_[0]}};
}

std::string Tensor::ToString() const {
  return "tensor(" + ToStringHelper() + ", dtype=" + dtype_.name() + ")";
}

const DataType& Tensor::DefaultDataType() { return kFloat32; }

Tensor::Strides Tensor::DefaultStrides(const Shape& shape) {
  if (shape.empty()) {
    return {};
  }

  Strides strides(shape.size());

  strides.back() = 1;

  for (auto i{shape.size() - 2}; i != -1; --i) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  return strides;
}

std::string Tensor::ToStringHelper() const {
  if (ndim() == 0) {
    if (dtype_ == kFloat32) {
      return std::to_string(*static_cast<float*>(data_));
    }

    // TODO: Handle more data types here.

    assert(false && "string conversion not implemented for this data type");
  }

  std::string result{"["};

  for (auto i{Index{0}}; i < shape_[0]; ++i) {
    result += operator[](i).ToStringHelper() + ", ";
  }

  result.pop_back();
  result.back() = ']';

  return result;
}

}  // namespace infini::ops
