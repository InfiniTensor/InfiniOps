#ifndef INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_H_
#define INFINI_OPS_CUDA_ADD_RMS_NORM_KERNEL_H_

#include <cassert>
#include <cstdint>

#include "base/add.h"
#include "base/add_rms_norm.h"
#include "base/rms_norm.h"
#include "cuda/add_rms_norm/kernel.cuh"
#include "cuda/runtime_utils.h"
#include "data_type.h"
#include "dispatcher.h"

namespace infini::ops {

template <typename Backend>
class CudaAddRmsNorm : public AddRmsNorm {
 public:
  CudaAddRmsNorm(const Tensor x1, const Tensor x2, const Tensor gamma,
                 float eps, Tensor y_out, Tensor x_out)
      : AddRmsNorm(x1, x2, gamma, eps, y_out, x_out),
        add_(x1, x2, x_out),
        rms_norm_(x_out, gamma, eps, y_out) {}

  void operator()(const Tensor x1, const Tensor x2, const Tensor gamma,
                  float eps, Tensor y_out, Tensor x_out) const override {
    add_.set_handle(handle_);
    add_.set_config(config_);
    add_.set_stream(stream_);
    add_.set_workspace(workspace_);
    add_.set_workspace_size_in_bytes(workspace_size_in_bytes_);
    add_(x1, x2, x_out);

    rms_norm_.set_handle(handle_);
    rms_norm_.set_config(config_);
    rms_norm_.set_stream(stream_);
    rms_norm_.set_workspace(workspace_);
    rms_norm_.set_workspace_size_in_bytes(workspace_size_in_bytes_);
    rms_norm_(x_out, gamma, eps, y_out);
  }

 private:
  mutable Operator<Add, Backend::kDeviceType> add_;

  mutable Operator<RmsNorm, Backend::kDeviceType> rms_norm_;
};

template <typename Backend>
class CudaAddRmsNormFused : public AddRmsNorm {
 public:
  CudaAddRmsNormFused(const Tensor x1, const Tensor x2, const Tensor gamma,
                      float eps, Tensor y_out, Tensor x_out)
      : AddRmsNorm(x1, x2, gamma, eps, y_out, x_out),
        x1_strides_{x1.strides()},
        x2_strides_{x2.strides()},
        y_out_strides_{y_out.strides()},
        x_out_strides_{x_out.strides()},
        stride_x1_batch_{x1_strides_.size() > 1 ? x1_strides_[0] : 0},
        stride_x1_nhead_{x1_strides_.size() > 1 ? x1_strides_[1]
                                                : x1_strides_[0]},
        stride_x2_batch_{x2_strides_.size() > 1 ? x2_strides_[0] : 0},
        stride_x2_nhead_{x2_strides_.size() > 1 ? x2_strides_[1]
                                                : x2_strides_[0]},
        stride_y_out_batch_{y_out_strides_.size() > 1 ? y_out_strides_[0] : 0},
        stride_y_out_nhead_{y_out_strides_.size() > 1 ? y_out_strides_[1]
                                                      : y_out_strides_[0]},
        stride_x_out_batch_{x_out_strides_.size() > 1 ? x_out_strides_[0] : 0},
        stride_x_out_nhead_{x_out_strides_.size() > 1 ? x_out_strides_[1]
                                                      : x_out_strides_[0]} {
    assert(x1.dtype() == x2.dtype());
    assert(x1.dtype() == gamma.dtype());
    assert(x1.dtype() == y_out.dtype());
    assert(x1.dtype() == x_out.dtype());
  }

  void operator()(const Tensor x1, const Tensor x2, const Tensor gamma,
                  float eps, Tensor y_out, Tensor x_out) const override {
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);

    uint32_t num_blocks = static_cast<uint32_t>(batch_size_ * nhead_);
    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 AllCudaBlockSizes>(
        {static_cast<int64_t>(y_out.dtype()), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          AddRmsNormKernel<kBlockSize, Backend::kDeviceType, float, T, T>
              <<<num_blocks, kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(y_out.data()), stride_y_out_batch_,
                  stride_y_out_nhead_, reinterpret_cast<T*>(x_out.data()),
                  stride_x_out_batch_, stride_x_out_nhead_,
                  reinterpret_cast<const T*>(x1.data()), stride_x1_batch_,
                  stride_x1_nhead_, reinterpret_cast<const T*>(x2.data()),
                  stride_x2_batch_, stride_x2_nhead_,
                  reinterpret_cast<const T*>(gamma.data()), nhead_, dim_, eps);
        },
        "CudaAddRmsNormFused::operator()");
  }

 private:
  Tensor::Strides x1_strides_;

  Tensor::Strides x2_strides_;

  Tensor::Strides y_out_strides_;

  Tensor::Strides x_out_strides_;

  int64_t stride_x1_batch_{0};

  int64_t stride_x1_nhead_{0};

  int64_t stride_x2_batch_{0};

  int64_t stride_x2_nhead_{0};

  int64_t stride_y_out_batch_{0};

  int64_t stride_y_out_nhead_{0};

  int64_t stride_x_out_batch_{0};

  int64_t stride_x_out_nhead_{0};
};

}  // namespace infini::ops

#endif
