#ifndef INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_H_

#include <cassert>
#include <cstdint>

#include "base/add_rms_norm.h"
#include "cuda/add_rms_norm/kernel.cuh"
#include "cuda/kernel_commons.cuh"
#include "cuda/runtime_utils.h"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

template <typename Backend>
class CudaAddRmsNorm : public AddRmsNorm {
 public:
  using AddRmsNorm::AddRmsNorm;

  void operator()(const Tensor input, const Tensor other, const Tensor weight,
                  float eps, Tensor out,
                  Tensor residual_out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    auto ndim = input_shape_.size();
    auto batch_size = ndim == 2 ? input_shape_[0] : input_shape_[0];
    auto nhead = ndim == 2 ? 1 : input_shape_[1];
    auto dim = input_shape_[ndim - 1];

    auto stride_input_batch =
        input_strides_.size() > 1 ? input_strides_[0] : 0;
    auto stride_input_nhead =
        input_strides_.size() > 1 ? input_strides_[1] : input_strides_[0];
    auto stride_other_batch =
        other_strides_.size() > 1 ? other_strides_[0] : 0;
    auto stride_other_nhead =
        other_strides_.size() > 1 ? other_strides_[1] : other_strides_[0];
    auto stride_out_batch = out_strides_.size() > 1 ? out_strides_[0] : 0;
    auto stride_out_nhead =
        out_strides_.size() > 1 ? out_strides_[1] : out_strides_[0];
    auto stride_residual_out_batch =
        residual_out_strides_.size() > 1 ? residual_out_strides_[0] : 0;
    auto stride_residual_out_nhead =
        residual_out_strides_.size() > 1 ? residual_out_strides_[1]
                                         : residual_out_strides_[0];

    uint32_t num_blocks = static_cast<uint32_t>(batch_size * nhead);

    assert(out.dtype() == input.dtype() && out.dtype() == other.dtype() &&
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
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<T*>(residual_out.data()),
                  stride_out_batch, stride_out_nhead,
                  stride_residual_out_batch, stride_residual_out_nhead,
                  reinterpret_cast<const T*>(input.data()), stride_input_batch,
                  stride_input_nhead, reinterpret_cast<const T*>(other.data()),
                  stride_other_batch, stride_other_nhead,
                  reinterpret_cast<const T*>(weight.data()), nhead, dim, eps);
        },
        "CudaAddRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
