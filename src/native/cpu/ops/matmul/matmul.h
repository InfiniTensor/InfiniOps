#ifndef INFINI_OPS_CPU_MATMUL_H_
#define INFINI_OPS_CPU_MATMUL_H_

#include <array>
#include <cassert>
#include <cstddef>
#include <utility>

#include "base/matmul.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cpu/caster_.h"

namespace infini::ops {

namespace cpu_matmul_detail {

inline constexpr std::size_t kMaxMatmulRank = 16;

inline std::size_t TensorBatchRank(std::size_t rank) {
  return rank > 1 ? rank - 2 : 0;
}

inline std::size_t ResultRank(std::size_t input_rank, std::size_t other_rank) {
  if (input_rank == 1 && other_rank == 1) return 0;
  if (input_rank == 1 || other_rank == 1) return 1;
  return 2;
}

inline std::size_t OutputOffset(
    const std::array<std::size_t, kMaxMatmulRank>& coordinates,
    const Tensor::Strides& strides, std::size_t rank) {
  std::size_t offset = 0;
  for (std::size_t axis = 0; axis < rank; ++axis) {
    offset += coordinates[axis] * strides[axis];
  }
  return offset;
}

inline std::size_t BatchOffset(
    const std::array<std::size_t, kMaxMatmulRank>& output_coordinates,
    const Tensor::Shape& shape, const Tensor::Strides& strides,
    std::size_t rank, std::size_t batch_rank) {
  const std::size_t tensor_batch_rank = TensorBatchRank(rank);
  std::size_t offset = 0;

  for (std::size_t index = 0; index < batch_rank; ++index) {
    const std::size_t output_axis = batch_rank - 1 - index;
    std::size_t coordinate = output_coordinates[output_axis];

    if (index < tensor_batch_rank) {
      const std::size_t axis = tensor_batch_rank - 1 - index;
      if (shape[axis] == 1) coordinate = 0;
      offset += coordinate * strides[axis];
    }
  }

  return offset;
}

inline void CoordinatesFromFlatIndex(
    std::size_t flat_index, const Tensor::Shape& shape,
    std::array<std::size_t, kMaxMatmulRank>& coordinates) {
  for (std::size_t axis = shape.size(); axis-- > 0;) {
    coordinates[axis] = flat_index % shape[axis];
    flat_index /= shape[axis];
  }
}

}  // namespace cpu_matmul_detail

template <>
class Operator<Matmul, Device::Type::kCpu> : public Matmul,
                                             Caster<Device::Type::kCpu> {
 public:
  Operator(const Tensor input, const Tensor other, Tensor out)
      : Matmul{input, other, out} {
    using namespace cpu_matmul_detail;
    assert(input_shape_.size() <= kMaxMatmulRank &&
           "operator `Matmul` supports input rank at most 16");
    assert(other_shape_.size() <= kMaxMatmulRank &&
           "operator `Matmul` supports other rank at most 16");
    assert(out_shape_.size() <= kMaxMatmulRank &&
           "operator `Matmul` supports output rank at most 16");
  }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    DispatchFunc<Device::Type::kCpu, AllFloatTypes>(
        out_type_,
        [&](auto tag) {
          using T = typename decltype(tag)::type;
          Compute<T>(input, other, out);
        },
        "`Operator<Matmul, Device::Type::kCpu>::operator()`");
  }

 private:
  template <typename T>
  void Compute(const Tensor input, const Tensor other, Tensor out) const {
    using namespace cpu_matmul_detail;

    const std::size_t input_rank = input_shape_.size();
    const std::size_t other_rank = other_shape_.size();
    const std::size_t result_rank = ResultRank(input_rank, other_rank);
    const std::size_t batch_rank = out_shape_.size() - result_rank;
    std::size_t output_size = 1;
    for (auto dimension : out_shape_) output_size *= dimension;

    const auto* input_data = static_cast<const T*>(input.data());
    const auto* other_data = static_cast<const T*>(other.data());
    auto* out_data = static_cast<T*>(out.data());

#pragma omp parallel for
    for (std::size_t output_index = 0; output_index < output_size;
         ++output_index) {
      std::array<std::size_t, kMaxMatmulRank> coordinates{};
      CoordinatesFromFlatIndex(output_index, out_shape_, coordinates);

      std::size_t input_offset = BatchOffset(
          coordinates, input_shape_, input_strides_, input_rank, batch_rank);
      std::size_t other_offset = BatchOffset(
          coordinates, other_shape_, other_strides_, other_rank, batch_rank);

      std::size_t input_coordinate = 0;
      std::size_t other_coordinate = 0;
      if (result_rank == 2) {
        input_coordinate = coordinates[batch_rank];
        other_coordinate = coordinates[batch_rank + 1];
      } else if (input_rank > 1) {
        input_coordinate = coordinates[batch_rank];
      } else {
        other_coordinate = coordinates[batch_rank];
      }

      if (result_rank == 2 || input_rank > 1) {
        input_offset += input_coordinate * input_strides_[input_rank - 2];
      }
      if (result_rank == 2 || other_rank > 1) {
        other_offset += other_coordinate * other_strides_[other_rank - 1];
      }

      const std::size_t contraction_size =
          input_rank == 1 ? input_shape_[0] : input_shape_[input_rank - 1];
      float sum = 0.0F;

      for (std::size_t contraction_index = 0;
           contraction_index < contraction_size; ++contraction_index) {
        std::size_t input_element_offset = input_offset;
        std::size_t other_element_offset = other_offset;

        input_element_offset +=
            contraction_index * input_strides_[input_rank - 1];
        if (other_rank > 1) {
          other_element_offset +=
              contraction_index * other_strides_[other_rank - 2];
        } else {
          other_element_offset += contraction_index * other_strides_[0];
        }

        sum += Cast<float>(input_data[input_element_offset]) *
               Cast<float>(other_data[other_element_offset]);
      }

      out_data[OutputOffset(coordinates, out_strides_, out_shape_.size())] =
          Cast<T>(sum);
    }
  }
};

}  // namespace infini::ops

#endif
