#ifndef INFINI_OPS_CPU_TEMPLATES_UNARY_ELEMENTWISE_H_
#define INFINI_OPS_CPU_TEMPLATES_UNARY_ELEMENTWISE_H_

#include <cstddef>

#include "common/generic_utils.h"
#include "cpu/caster_.h"
#include "dispatcher.h"
#include "tensor.h"

namespace infini::ops {

// CPU unary elementwise brick with dual-dtype dispatch.
//
// `Op` is a host-side functor called as `op.template operator()<TIn, TOut>(x)`,
// allowing the functor to know both input and output types.  Handles
// non-contiguous tensors via `IndexToOffset`.
template <typename InputTypeList, typename OutputTypeList, typename Op>
void CpuUnaryElementwise(const Tensor in, Tensor out,
                          Tensor::Size output_size, Tensor::Size ndim,
                          bool in_contig, bool out_contig,
                          const Tensor::Shape& in_shape,
                          const Tensor::Shape& out_shape,
                          const Tensor::Strides& in_strides,
                          const Tensor::Strides& out_strides,
                          DataType input_dtype, DataType output_dtype, Op op) {
  DispatchFunc<Device::Type::kCpu, InputTypeList>(
      input_dtype,
      [&](auto in_tag) {
        using TIn = typename decltype(in_tag)::type;

        DispatchFunc<Device::Type::kCpu, OutputTypeList>(
            output_dtype,
            [&](auto out_tag) {
              using TOut = typename decltype(out_tag)::type;

              const auto* in_ptr = static_cast<const TIn*>(in.data());
              auto* out_ptr = static_cast<TOut*>(out.data());

#pragma omp parallel for
              for (Tensor::Size i = 0; i < output_size; ++i) {
                auto ii = in_contig
                              ? i
                              : utils::IndexToOffset(i, ndim, in_shape.data(),
                                                     in_strides.data());
                auto oi = out_contig
                              ? i
                              : utils::IndexToOffset(i, ndim, out_shape.data(),
                                                     out_strides.data());

                out_ptr[oi] = op.template operator()<TIn, TOut>(in_ptr[ii]);
              }
            },
            "CpuUnaryElementwise (out)");
      },
      "CpuUnaryElementwise (in)");
}

}  // namespace infini::ops

#endif
