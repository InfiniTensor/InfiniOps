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

  void operator()(const Tensor x1, const Tensor x2, const Tensor weight,
                  float eps, Tensor y_out, Tensor x_out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    auto stride_x1_batch = x1_strides_.size() > 1 ? x1_strides_[0] : 0;
    auto stride_x1_nhead =
        x1_strides_.size() > 1 ? x1_strides_[1] : x1_strides_[0];
    auto stride_x2_batch = x2_strides_.size() > 1 ? x2_strides_[0] : 0;
    auto stride_x2_nhead =
        x2_strides_.size() > 1 ? x2_strides_[1] : x2_strides_[0];
    auto stride_y_out_batch =
        y_out_strides_.size() > 1 ? y_out_strides_[0] : 0;
    auto stride_y_out_nhead =
        y_out_strides_.size() > 1 ? y_out_strides_[1] : y_out_strides_[0];
    auto stride_x_out_batch =
        x_out_strides_.size() > 1 ? x_out_strides_[0] : 0;
    auto stride_x_out_nhead =
        x_out_strides_.size() > 1 ? x_out_strides_[1] : x_out_strides_[0];

    uint32_t num_blocks = static_cast<uint32_t>(batch_size_ * nhead_);

    assert(x1.dtype() == x2.dtype() && x1.dtype() == weight.dtype() &&
           x1.dtype() == y_out.dtype() && x1.dtype() == x_out.dtype());

    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 AllCudaBlockSizes>(
        {static_cast<int64_t>(y_out.dtype()), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          size_t smem_bytes = dim_ * sizeof(float);

          AddRmsNormKernel<kBlockSize, Backend::kDeviceType, float, T, T>
              <<<num_blocks, kBlockSize, smem_bytes, cuda_stream>>>(
                  reinterpret_cast<T*>(y_out.data()), stride_y_out_batch,
                  stride_y_out_nhead, reinterpret_cast<T*>(x_out.data()),
                  stride_x_out_batch, stride_x_out_nhead,
                  reinterpret_cast<const T*>(x1.data()), stride_x1_batch,
                  stride_x1_nhead, reinterpret_cast<const T*>(x2.data()),
                  stride_x2_batch, stride_x2_nhead,
                  reinterpret_cast<const T*>(weight.data()), nhead_, dim_,
                  eps_);
        },
        "CudaAddRmsNorm::operator()");
  }
};

}  // namespace infini::ops

#endif
