#ifndef INFINI_OPS_CUDA_MATMUL_KERNEL_H_
#define INFINI_OPS_CUDA_MATMUL_KERNEL_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "base/matmul.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/caster.cuh"

namespace infini::ops {

namespace cuda_matmul_detail {

inline constexpr std::size_t kMaxMatmulRank = 16;
inline constexpr unsigned int kMatmulBlockSize = 256;

struct Layout {
  std::size_t input_shape[kMaxMatmulRank]{};
  std::size_t other_shape[kMaxMatmulRank]{};
  std::size_t out_shape[kMaxMatmulRank]{};
  std::ptrdiff_t input_strides[kMaxMatmulRank]{};
  std::ptrdiff_t other_strides[kMaxMatmulRank]{};
  std::ptrdiff_t out_strides[kMaxMatmulRank]{};
  std::size_t input_rank{};
  std::size_t other_rank{};
  std::size_t out_rank{};
  std::size_t batch_rank{};
  std::size_t result_rank{};
  std::size_t contraction_size{};
  std::size_t output_size{};

  __device__ std::size_t InputOffset(const std::size_t* coordinates,
                                     std::size_t contraction_index) const {
    std::size_t offset =
        TensorOffset(coordinates, input_shape, input_strides, input_rank);
    std::size_t coordinate = 0;

    if (result_rank == 2 || input_rank > 1) {
      coordinate = result_rank == 0 ? 0 : coordinates[batch_rank];
      offset += coordinate * input_strides[input_rank - 2];
    }
    offset += contraction_index * input_strides[input_rank - 1];
    return offset;
  }

  __device__ std::size_t OtherOffset(const std::size_t* coordinates,
                                     std::size_t contraction_index) const {
    std::size_t offset =
        TensorOffset(coordinates, other_shape, other_strides, other_rank);
    std::size_t coordinate = 0;

    if (result_rank == 2 || other_rank > 1) {
      coordinate = coordinates[batch_rank + result_rank - 1];
      offset += coordinate * other_strides[other_rank - 1];
      offset += contraction_index * other_strides[other_rank - 2];
    } else {
      offset += contraction_index * other_strides[0];
    }
    return offset;
  }

  __device__ std::size_t OutputOffset(const std::size_t* coordinates) const {
    std::size_t offset = 0;
    for (std::size_t axis = 0; axis < out_rank; ++axis) {
      offset += coordinates[axis] * out_strides[axis];
    }
    return offset;
  }

 private:
  __device__ std::size_t TensorOffset(const std::size_t* coordinates,
                                      const std::size_t* shape,
                                      const std::ptrdiff_t* strides,
                                      std::size_t rank) const {
    const std::size_t tensor_batch_rank = rank > 1 ? rank - 2 : 0;
    std::size_t offset = 0;

    for (std::size_t index = 0; index < batch_rank; ++index) {
      const std::size_t output_axis = batch_rank - 1 - index;
      std::size_t coordinate = coordinates[output_axis];
      if (index < tensor_batch_rank) {
        const std::size_t axis = tensor_batch_rank - 1 - index;
        if (shape[axis] == 1) coordinate = 0;
        offset += coordinate * strides[axis];
      }
    }
    return offset;
  }
};

}  // namespace cuda_matmul_detail

template <Device::Type kDev, typename T>
__global__ void MatmulKernel(T* __restrict__ out, const T* __restrict__ input,
                             const T* __restrict__ other,
                             cuda_matmul_detail::Layout layout) {
  namespace detail = cuda_matmul_detail;

  const std::size_t output_index = blockIdx.x;
  if (output_index >= layout.output_size) return;

  float sum = 0.0F;
  std::size_t coordinates[detail::kMaxMatmulRank]{};
  std::size_t flat_index = output_index;
  for (std::size_t axis = layout.out_rank; axis-- > 0;) {
    coordinates[axis] = flat_index % layout.out_shape[axis];
    flat_index /= layout.out_shape[axis];
  }

  for (std::size_t contraction_index = threadIdx.x;
       contraction_index < layout.contraction_size;
       contraction_index += blockDim.x) {
    const auto input_value = Caster<kDev>::template Cast<float>(
        input[layout.InputOffset(coordinates, contraction_index)]);
    const auto other_value = Caster<kDev>::template Cast<float>(
        other[layout.OtherOffset(coordinates, contraction_index)]);
    sum += input_value * other_value;
  }

  __shared__ float partial[detail::kMatmulBlockSize];
  partial[threadIdx.x] = sum;
  __syncthreads();

  for (std::size_t stride = blockDim.x / 2; stride > 0; stride /= 2) {
    if (threadIdx.x < stride) {
      partial[threadIdx.x] += partial[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    out[layout.OutputOffset(coordinates)] =
        Caster<kDev>::template Cast<T>(partial[0]);
  }
}

template <typename Backend>
class CudaMatmul : public Matmul {
 public:
  CudaMatmul(const Tensor input, const Tensor other, Tensor out)
      : Matmul{input, other, out} {
    namespace detail = cuda_matmul_detail;

    assert(input_shape_.size() <= detail::kMaxMatmulRank &&
           "operator `Matmul` supports input rank at most 16");
    assert(other_shape_.size() <= detail::kMaxMatmulRank &&
           "operator `Matmul` supports other rank at most 16");
    assert(out_shape_.size() <= detail::kMaxMatmulRank &&
           "operator `Matmul` supports output rank at most 16");

    layout_.input_rank = input_shape_.size();
    layout_.other_rank = other_shape_.size();
    layout_.out_rank = out_shape_.size();
    std::copy(input_shape_.begin(), input_shape_.end(), layout_.input_shape);
    std::copy(other_shape_.begin(), other_shape_.end(), layout_.other_shape);
    std::copy(out_shape_.begin(), out_shape_.end(), layout_.out_shape);
    std::copy(input_strides_.begin(), input_strides_.end(),
              layout_.input_strides);
    std::copy(other_strides_.begin(), other_strides_.end(),
              layout_.other_strides);
    std::copy(out_strides_.begin(), out_strides_.end(), layout_.out_strides);

    if (layout_.input_rank == 1 && layout_.other_rank == 1) {
      layout_.result_rank = 0;
    } else if (layout_.input_rank == 1 || layout_.other_rank == 1) {
      layout_.result_rank = 1;
    } else {
      layout_.result_rank = 2;
    }
    layout_.batch_rank = layout_.out_rank - layout_.result_rank;
    layout_.contraction_size =
        layout_.input_rank == 1 ? layout_.input_shape[0]
                                : layout_.input_shape[layout_.input_rank - 1];
    layout_.output_size = 1;
    for (std::size_t axis = 0; axis < layout_.out_rank; ++axis) {
      layout_.output_size *= layout_.out_shape[axis];
    }
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    auto stream = static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    DispatchFunc<Backend::kDeviceType, AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          MatmulKernel<Backend::kDeviceType, T>
              <<<static_cast<unsigned int>(layout_.output_size),
                 cuda_matmul_detail::kMatmulBlockSize, 0, stream>>>(
                  static_cast<T*>(out.data()),
                  static_cast<const T*>(input.data()),
                  static_cast<const T*>(other.data()), layout_);
        },
        "`CudaMatmul::operator()`");
  }

 private:
  cuda_matmul_detail::Layout layout_;
};

}  // namespace infini::ops

#endif
