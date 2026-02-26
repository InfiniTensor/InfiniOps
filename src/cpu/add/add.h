#ifndef INFINI_OPS_CPU_ADD_ADD_H_
#define INFINI_OPS_CPU_ADD_ADD_H_

#include <utility>

#include "base/add.h"
#include "common/generic_utils.h"

namespace infini::ops {

template <>
class Operator<Add, Device::Type::kCpu> : public Add {
 public:
  Operator(const Tensor a, const Tensor b, Tensor c) : Add{a, b, c} {
    // TODO: Check constraints.
  }

  void operator()(void* stream, const Tensor a, const Tensor b,
                  Tensor c) const override {
    DispatchFunc<ConcatType<FloatTypes, AllIntTypes>>(
        c_type_, [&]<typename T>() { compute<T>(stream, a, b, c); },
        "Operator<Add, Device::Type::kCpu>::operator()");
  }

 private:
  template <typename T>
  void compute(void* stream, const Tensor a, const Tensor b, Tensor c) const {
    const auto* a_ptr = static_cast<const T*>(a.data());
    const auto* b_ptr = static_cast<const T*>(b.data());
    auto* c_ptr = static_cast<T*>(c.data());

    auto get_idx = [&](Tensor::Size i, bool is_contig, const auto* shape,
                       const auto* strides) {
      return is_contig ? i : utils::indexToOffset(i, ndim_, shape, strides);
    };

#pragma omp parallel for
    for (Tensor::Size i = 0; i < output_size_; ++i) {
      auto a_idx =
          get_idx(i, is_a_contiguous_, a_shape_.data(), a_strides_.data());
      auto b_idx =
          get_idx(i, is_b_contiguous_, b_shape_.data(), b_strides_.data());
      auto c_idx =
          get_idx(i, is_c_contiguous_, c_shape_.data(), c_strides_.data());

      c_ptr[c_idx] = a_ptr[a_idx] + b_ptr[b_idx];
    }
  }
};

}  // namespace infini::ops

#endif
