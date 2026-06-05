#ifndef INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_H_

#include <cassert>
#include <cstdint>

#include "base/add_rms_norm.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/add_rms_norm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaAddRmsNorm : public AddRmsNorm {
 public:
  using AddRmsNorm::AddRmsNorm;

  void operator()(const Tensor input, const Tensor residual,
                  const Tensor weight, float eps, Tensor out,
                  Tensor residual_out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    auto stride_input_batch = input_strides_.size() > 1 ? input_strides_[0] : 0;
    auto stride_input_nhead =
        input_strides_.size() > 1 ? input_strides_[1] : input_strides_[0];
    auto stride_residual_batch =
        residual_strides_.size() > 1 ? residual_strides_[0] : 0;
    auto stride_residual_nhead = residual_strides_.size() > 1
                                     ? residual_strides_[1]
                                     : residual_strides_[0];
    auto stride_out_batch = out_strides_.size() > 1 ? out_strides_[0] : 0;
    auto stride_out_nhead =
        out_strides_.size() > 1 ? out_strides_[1] : out_strides_[0];
    auto stride_residual_out_batch =
        residual_out_strides_.size() > 1 ? residual_out_strides_[0] : 0;
    auto stride_residual_out_nhead = residual_out_strides_.size() > 1
                                         ? residual_out_strides_[1]
                                         : residual_out_strides_[0];

    uint32_t num_blocks = static_cast<uint32_t>(batch_size_ * nhead_);

    assert(out.dtype() == input.dtype() && out.dtype() == residual.dtype() &&
           out.dtype() == weight.dtype() &&
           out.dtype() == residual_out.dtype());

    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 AllCudaBlockSizes>(
        {static_cast<int64_t>(out.dtype()), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          AddRmsNormKernel<kBlockSize, Backend::kDeviceType, float, T, T>
              <<<num_blocks, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()), stride_out_batch,
                  stride_out_nhead, reinterpret_cast<T*>(residual_out.data()),
                  stride_residual_out_batch, stride_residual_out_nhead,
                  reinterpret_cast<const T*>(input.data()), stride_input_batch,
                  stride_input_nhead,
                  reinterpret_cast<const T*>(residual.data()),
                  stride_residual_batch, stride_residual_nhead,
                  reinterpret_cast<const T*>(weight.data()), nhead_, dim_, eps);
        },
        "CudaAddRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
