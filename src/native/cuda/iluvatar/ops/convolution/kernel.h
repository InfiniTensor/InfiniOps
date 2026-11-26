#ifndef INFINI_OPS_ILUVATAR_CONVOLUTION_KERNEL_H_
#define INFINI_OPS_ILUVATAR_CONVOLUTION_KERNEL_H_

#include <memory>
#include <utility>

#ifdef WITH_ILUVATAR_CUDNN
#include "native/cuda/iluvatar/ops/convolution/cudnn.h"
#endif

#include "native/cuda/iluvatar/caster.cuh"
#include "native/cuda/iluvatar/runtime_.h"
#include "native/cuda/ops/convolution/kernel.h"

namespace infini::ops {

template <>
class Operator<Convolution, Device::Type::kIluvatar>
    : public CudaConv<Runtime<Device::Type::kIluvatar>, Convolution> {
 public:
  using Base = CudaConv<Runtime<Device::Type::kIluvatar>, Convolution>;

#ifndef WITH_ILUVATAR_CUDNN
  using Base::Base;
#else
  Operator(const Tensor input, const Tensor weight, std::optional<Tensor> bias,
           const std::vector<int64_t> stride,
           const std::vector<int64_t> padding,
           const std::vector<int64_t> dilation, bool transposed,
           const std::vector<int64_t> output_padding, int64_t groups,
           Tensor out)
      : Base(input, weight, bias, stride, padding, dilation, transposed,
             output_padding, groups, out) {
    if (iluvatar_conv_detail::CudnnState::Eligible(
            input, weight, bias, stride, padding, dilation, transposed,
            output_padding, groups, out)) {
      auto candidate = std::make_unique<iluvatar_conv_detail::CudnnState>();
      if (candidate->Initialize(input, weight, stride, padding, out))
        state_ = std::move(candidate);
    }
  }

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias, const std::vector<int64_t> stride,
                  const std::vector<int64_t> padding,
                  const std::vector<int64_t> dilation, bool transposed,
                  const std::vector<int64_t> output_padding, int64_t groups,
                  Tensor out) const override {
    if (state_ && state_->Run(input, weight, stream_, out)) return;
    Base::operator()(input, weight, bias, stride, padding, dilation, transposed,
                     output_padding, groups, out);
  }

 private:
  std::unique_ptr<iluvatar_conv_detail::CudnnState> state_;
#endif
};

}  // namespace infini::ops

#endif
