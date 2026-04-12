#ifndef INFINI_OPS_CPU_TEMPLATES_BINARY_ELEMENTWISE_H_
#define INFINI_OPS_CPU_TEMPLATES_BINARY_ELEMENTWISE_H_

#include <cstddef>

#include "common/generic_utils.h"
#include "cpu/caster_.h"
#include "dispatcher.h"
#include "tensor.h"

namespace infini::ops {

// CPU binary elementwise brick.
//
// `Op` is a host-side functor: `T operator()(const T&, const T&) const`.
// Handles non-contiguous tensors via `IndexToOffset` and promotes FP16/BF16
// to float for computation.
template <typename TypeList, typename Op>
void CpuBinaryElementwise(const Tensor a, const Tensor b, Tensor out,
                           Tensor::Size output_size, Tensor::Size ndim,
                           bool a_contig, bool b_contig, bool out_contig,
                           const Tensor::Shape& a_shape,
                           const Tensor::Shape& b_shape,
                           const Tensor::Shape& out_shape,
                           const Tensor::Strides& a_strides,
                           const Tensor::Strides& b_strides,
                           const Tensor::Strides& out_strides, DataType dtype,
                           Op op) {
  DispatchFunc<Device::Type::kCpu, TypeList>(
      dtype,
      [&](auto tag) {
        using T = typename decltype(tag)::type;
        using ComputeType =
            std::conditional_t<IsBFloat16<Device::Type::kCpu, T> ||
                                   IsFP16<Device::Type::kCpu, T>,
                               float, T>;

        const auto* a_ptr = static_cast<const T*>(a.data());
        const auto* b_ptr = static_cast<const T*>(b.data());
        auto* out_ptr = static_cast<T*>(out.data());

#pragma omp parallel for
        for (Tensor::Size i = 0; i < output_size; ++i) {
          auto ai = a_contig
                        ? i
                        : utils::IndexToOffset(i, ndim, a_shape.data(),
                                               a_strides.data());
          auto bi = b_contig
                        ? i
                        : utils::IndexToOffset(i, ndim, b_shape.data(),
                                               b_strides.data());
          auto oi = out_contig
                        ? i
                        : utils::IndexToOffset(i, ndim, out_shape.data(),
                                               out_strides.data());

          out_ptr[oi] = Caster<Device::Type::kCpu>::Cast<T>(
              op(Caster<Device::Type::kCpu>::Cast<ComputeType>(a_ptr[ai]),
                 Caster<Device::Type::kCpu>::Cast<ComputeType>(b_ptr[bi])));
        }
      },
      "CpuBinaryElementwise");
}

}  // namespace infini::ops

#endif
