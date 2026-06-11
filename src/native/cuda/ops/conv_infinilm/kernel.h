#ifndef INFINI_OPS_CUDA_CONV_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_CONV_INFINILM_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <optional>
#include <vector>

#include "base/conv_infinilm.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/conv_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaConvInfinilm : public ConvInfinilm {
 public:
  CudaConvInfinilm(const Tensor input, const Tensor weight,
                   std::optional<Tensor> bias,
                   const std::vector<int64_t> padding,
                   const std::vector<int64_t> stride,
                   const std::vector<int64_t> dilation, const int64_t groups,
                   Tensor out)
      : ConvInfinilm{input,  weight,   bias,   padding,
                     stride, dilation, groups, out} {
    size_t input_shape_size = input_shape_.size() * sizeof(*d_input_shape_);
    size_t weight_shape_size = weight_shape_.size() * sizeof(*d_weight_shape_);
    size_t out_shape_size = out_shape_.size() * sizeof(*d_out_shape_);
    size_t input_strides_size =
        input_strides_.size() * sizeof(*d_input_strides_);
    size_t weight_strides_size =
        weight_strides_.size() * sizeof(*d_weight_strides_);
    size_t out_strides_size = out_strides_.size() * sizeof(*d_out_strides_);
    size_t bias_strides_size = sizeof(*d_bias_strides_);
    size_t attrs_size = spatial_ndim_ * sizeof(*d_padding_);
    const size_t metadata_size = input_shape_size + weight_shape_size +
                                 out_shape_size + input_strides_size +
                                 weight_strides_size + out_strides_size +
                                 bias_strides_size + 3 * attrs_size;
    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

    size_t offset = 0;
    d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_shape_.data(),
                input_shape_size);
    offset += input_shape_size;

    d_weight_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, weight_shape_.data(),
                weight_shape_size);
    offset += weight_shape_size;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_shape_.data(), out_shape_size);
    offset += out_shape_size;

    d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_strides_.data(),
                input_strides_size);
    offset += input_strides_size;

    d_weight_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, weight_strides_.data(),
                weight_strides_size);
    offset += weight_strides_size;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_strides_.data(),
                out_strides_size);
    offset += out_strides_size;

    d_bias_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    Tensor::Stride bias_stride = has_bias_ ? bias_strides_[0] : 0;
    std::memcpy(metadata.data() + offset, &bias_stride, bias_strides_size);
    offset += bias_strides_size;

    d_padding_ = reinterpret_cast<int64_t*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, padding_.data(), attrs_size);
    offset += attrs_size;

    d_stride_ = reinterpret_cast<int64_t*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, stride_.data(), attrs_size);
    offset += attrs_size;

    d_dilation_ = reinterpret_cast<int64_t*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, dilation_.data(), attrs_size);

    Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::MemcpyHostToDevice);
  }

  ~CudaConvInfinilm() { Backend::Free(d_metadata_); }

  void operator()(const Tensor input, const Tensor weight,
                  std::optional<Tensor> bias,
                  const std::vector<int64_t> padding,
                  const std::vector<int64_t> stride,
                  const std::vector<int64_t> dilation, const int64_t groups,
                  Tensor out) const override {
    (void)padding;
    (void)stride;
    (void)dilation;
    (void)groups;
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);
    dim3 block(std::min(static_cast<Tensor::Size>(block_size), output_size_));
    dim3 grid(utils::CeilDiv(output_size_, block.x));

    DispatchFunc<ConcatType<List<DataType::kFloat32>, ReducedFloatTypes>,
                 List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);
          const T* bias_ptr = bias.has_value()
                                  ? reinterpret_cast<const T*>(bias->data())
                                  : nullptr;

          ConvInfinilmKernel<Backend::kDeviceType, T, kBlockSize>
              <<<grid, block, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()),
                  reinterpret_cast<const T*>(weight.data()), bias_ptr,
                  d_input_shape_, d_weight_shape_, d_out_shape_,
                  d_input_strides_, d_weight_strides_, d_out_strides_,
                  d_bias_strides_, d_padding_, d_stride_, d_dilation_,
                  output_size_, spatial_ndim_, kernel_size_, groups_,
                  has_bias_);
        },
        "CudaConvInfinilm::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_weight_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_weight_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};

  Tensor::Stride* d_bias_strides_{nullptr};

  int64_t* d_padding_{nullptr};

  int64_t* d_stride_{nullptr};

  int64_t* d_dilation_{nullptr};
};

}  // namespace infini::ops

#endif
