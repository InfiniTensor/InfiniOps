#ifndef INFINI_OPS_CAMBRICON_CNNL_UTILS_H_
#define INFINI_OPS_CAMBRICON_CNNL_UTILS_H_

#include <cassert>
#include <cstdint>
#include <limits>
#include <memory>
#include <type_traits>
#include <vector>

#include "native/cambricon/common.h"
#include "tensor.h"

namespace infini::ops::cnnl_utils {

struct HandleDeleter {
  using pointer = cnnlHandle_t;

  void operator()(pointer handle) const noexcept {
    if (handle) {
      (void)cnnlDestroy(handle);
    }
  }
};

using Handle =
    std::unique_ptr<std::remove_pointer_t<cnnlHandle_t>, HandleDeleter>;

inline Handle CreateHandle() {
  cnnlHandle_t handle{nullptr};
  [[maybe_unused]] const auto status = cnnlCreate(&handle);
  assert(status == CNNL_STATUS_SUCCESS && "`cnnlCreate` failed.");

  return Handle{handle};
}

struct TensorDescriptorDeleter {
  using pointer = cnnlTensorDescriptor_t;

  void operator()(pointer desc) const noexcept {
    if (desc) {
      (void)cnnlDestroyTensorDescriptor(desc);
    }
  }
};

using TensorDescriptor =
    std::unique_ptr<std::remove_pointer_t<cnnlTensorDescriptor_t>,
                    TensorDescriptorDeleter>;

inline TensorDescriptor CreateTensorDescriptor() {
  cnnlTensorDescriptor_t desc{nullptr};
  [[maybe_unused]] const auto status = cnnlCreateTensorDescriptor(&desc);
  assert(status == CNNL_STATUS_SUCCESS &&
         "`cnnlCreateTensorDescriptor` failed.");

  return TensorDescriptor{desc};
}

namespace detail {

template <typename Integer>
int CheckedInt(Integer value) {
  static_assert(std::is_integral_v<Integer>);

  [[maybe_unused]] bool out_of_range{false};
  if constexpr (std::is_signed_v<Integer>) {
    const auto wide = static_cast<std::intmax_t>(value);
    out_of_range = wide < std::numeric_limits<int>::min() ||
                   wide > std::numeric_limits<int>::max();
  } else {
    const auto wide = static_cast<std::uintmax_t>(value);
    out_of_range =
        wide > static_cast<std::uintmax_t>(std::numeric_limits<int>::max());
  }

  assert(!out_of_range &&
         "`CNNL tensor descriptor` value does not fit in `int`.");

  return static_cast<int>(value);
}

template <typename Values>
std::vector<int> CheckedIntVector(const Values& values) {
  std::vector<int> result;
  result.reserve(values.size());
  for (const auto value : values) {
    result.push_back(CheckedInt(value));
  }
  return result;
}

}  // namespace detail

inline void SetTensorDescriptor(cnnlTensorDescriptor_t desc, DataType dtype,
                                const Tensor::Shape& shape,
                                const Tensor::Strides& strides) {
  assert(!shape.empty() && shape.size() == strides.size() &&
         "`CNNL tensor descriptor` requires matching non-empty shape and "
         "strides.");

  const auto cnnl_dtype = GetDataType(dtype);
  assert(cnnl_dtype != CNNL_DTYPE_INVALID &&
         "`CNNL tensor descriptor` does not support this data type.");

  const auto cnnl_shape = detail::CheckedIntVector(shape);
  const auto cnnl_strides = detail::CheckedIntVector(strides);
  const auto ndim = detail::CheckedInt(shape.size());

  [[maybe_unused]] const auto status =
      cnnlSetTensorDescriptorEx(desc, CNNL_LAYOUT_ARRAY, cnnl_dtype, ndim,
                                cnnl_shape.data(), cnnl_strides.data());
  assert(status == CNNL_STATUS_SUCCESS &&
         "`cnnlSetTensorDescriptorEx` failed.");
}

inline TensorDescriptor MakeTensorDescriptor(DataType dtype,
                                             const Tensor::Shape& shape,
                                             const Tensor::Strides& strides) {
  auto desc = CreateTensorDescriptor();
  SetTensorDescriptor(desc.get(), dtype, shape, strides);
  return desc;
}

}  // namespace infini::ops::cnnl_utils

#endif
