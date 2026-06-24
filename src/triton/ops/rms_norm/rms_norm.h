#ifndef INFINI_OPS_TRITON_RMS_NORM_H_
#define INFINI_OPS_TRITON_RMS_NORM_H_

#include <cuda.h>

#include <cassert>
#include <cstdint>

#include "base/rms_norm.h"
#include "data_type.h"
#include "rms_norm/infini_ops_triton_rms_norm.h"

namespace infini::ops {

template <>
class Operator<RmsNorm, Device::Type::kNvidia, 8> : public RmsNorm {
 public:
  using RmsNorm::operator();

  Operator(const Tensor input, const Tensor weight, float eps, Tensor out)
      : RmsNorm{input, weight, eps, out} {}

  Operator(const Tensor input, const Tensor weight, Tensor out)
      : RmsNorm{input, weight, out} {}

  void operator()(const Tensor input, const Tensor weight, float eps,
                  Tensor out) const override {
    assert(input.dtype() == out.dtype() &&
           "Triton `RmsNorm` requires input and output to have the same dtype");

    load_infini_ops_triton_rms_norm(out.dtype());

    const auto input_strides = input.strides();
    const auto weight_strides = weight.strides();
    const auto out_strides = out.strides();

    const auto n_rows = static_cast<int32_t>(batch_size_ * nhead_);
    const auto n_cols = static_cast<int32_t>(dim_);

    const auto stride_xm = static_cast<int64_t>(input_strides[ndim_ - 2]);
    const auto stride_xn = static_cast<int64_t>(input_strides[ndim_ - 1]);
    const auto stride_wn = static_cast<int64_t>(weight_strides.back());
    const auto stride_ym = static_cast<int64_t>(out_strides[ndim_ - 2]);
    const auto stride_yn = static_cast<int64_t>(out_strides[ndim_ - 1]);

    auto result = launch_infini_ops_triton_rms_norm(
        out.dtype(), static_cast<CUstream>(stream_),
        reinterpret_cast<CUdeviceptr>(const_cast<void*>(input.data())),
        reinterpret_cast<CUdeviceptr>(const_cast<void*>(weight.data())),
        reinterpret_cast<CUdeviceptr>(out.data()), eps, n_rows, n_cols,
        stride_xm, stride_xn, stride_wn, stride_ym, stride_yn);

    assert(result == CUDA_SUCCESS && "Triton `RmsNorm` launch failed");
  }
};

}  // namespace infini::ops

#endif
