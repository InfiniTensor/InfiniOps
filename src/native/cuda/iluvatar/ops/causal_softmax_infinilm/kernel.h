#ifndef INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CAUSAL_SOFTMAX_INFINILM_KERNEL_H_

#include <utility>

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/ops/causal_softmax_infinilm/kernel_v2.cuh"
#include "native/cuda/iluvatar/ops/causal_softmax_infinilm/policy.h"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/causal_softmax_infinilm/kernel.h"

namespace infini::ops {

template <>
class Operator<CausalSoftmaxInfinilm, Device::Type::kIluvatar>
    : public CudaCausalSoftmaxInfinilm<Runtime<Device::Type::kIluvatar>> {
  using Parent = CudaCausalSoftmaxInfinilm<Runtime<Device::Type::kIluvatar>>;

 public:
  using Parent::Parent;

  void operator()(const Tensor input, Tensor out) const override {
    if (input.ndim() == 3 && out.ndim() == 3 && input.shape() == out.shape() &&
        iluvatar::CanUseCausalSoftmaxFp32(
            input.dtype() == DataType::kBFloat16 &&
                out.dtype() == input.dtype(),
            input.device() == out.device(),
            input.stride(2) == 1 && out.stride(2) == 1 &&
                input.stride(1) == static_cast<Tensor::Stride>(input.size(2)) &&
                out.stride(1) == input.stride(1) &&
                input.stride(0) == static_cast<Tensor::Stride>(input.size(1) *
                                                               input.size(2)) &&
                out.stride(0) == input.stride(0),
            input.size(0), input.size(1), input.size(2))) {
      const std::size_t rows = input.size(0) * input.size(1);
      const unsigned gx = rows < 65535 ? static_cast<unsigned>(rows) : 65535;
      const unsigned gy = static_cast<unsigned>((rows + gx - 1) / gx);
      auto stream = static_cast<cudaStream_t>(stream_ ? stream_ : 0);
      iluvatar::CausalSoftmaxFp32Kernel<1024>
          <<<dim3(gx, gy), 1024, 0, stream>>>(
              static_cast<__nv_bfloat16*>(out.data()),
              static_cast<const __nv_bfloat16*>(input.data()), rows,
              input.size(1), input.size(2));
      return;
    }
    Parent::operator()(input, out);
  }
};

}  // namespace infini::ops

#endif
