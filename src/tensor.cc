#include "tensor.h"

#include <cassert>

#include "dispatcher.h"

namespace infini::ops {

Tensor::Tensor(void* data, std::initializer_list<Size> shape,
               const DataType& dtype, const Device& device,
               std::initializer_list<Stride> strides)
    : Tensor{data, decltype(shape_){shape}, dtype, device,
             decltype(strides_){strides}} {}

Tensor Tensor::operator[](const Index& index) const {
  return {reinterpret_cast<decltype(data_)>(
              reinterpret_cast<decltype(index)>(data_) +
              index * strides_[0] * element_size()),
          Shape{shape_.cbegin() + 1, shape_.cend()}, dtype_, device_,
          Strides{strides_.cbegin() + 1, strides_.cend()}};
}

void*& Tensor::data() { return data_; }

const void* const& Tensor::data() const { return data_; }

const Tensor::Shape& Tensor::shape() const { return shape_; }

const DataType& Tensor::dtype() const { return dtype_; }

const Device& Tensor::device() const { return device_; }

const Tensor::Strides& Tensor::strides() const { return strides_; }

Tensor::Size Tensor::size(const Index& index) const { return shape_[index]; }

Tensor::Stride Tensor::stride(const Index& index) const {
  return strides_[index];
}

Tensor::Size Tensor::ndim() const { return shape_.size(); }

Tensor::Size Tensor::element_size() const { return kDataTypeToSize.at(dtype_); }

Tensor Tensor::T() const {
  return {data_,
          {shape_[1], shape_[0]},
          dtype_,
          device_,
          {strides_[1], strides_[0]}};
}

std::string Tensor::ToString() const {
  return "tensor(" + ToStringHelper() +
         ", dtype=" + std::string(kDataTypeToDesc.at(dtype_)) + ", device='" +
         device_.ToString() + "')";
}

const DataType Tensor::DefaultDataType() { return DataType::kFloat32; }

Device Tensor::DefaultDevice() { return Device{Device::Type::kCpu}; }

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
    return DispatchFunc<FloatTypes>(
        dtype_,
        [&]<typename T>() { return std::to_string(*static_cast<T*>(data_)); },
        "ToStringHelper");
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
