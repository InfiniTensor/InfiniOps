#ifndef INFINI_OPS_CUDA_CAST_KERNEL_H_
#define INFINI_OPS_CUDA_CAST_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "base/cast.h"
#include "common/generic_utils.h"
#include "cuda/cast/kernel.cuh"
#include "cuda/kernel_commons.cuh"
#include "cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaCast : public Cast {
 public:
  CudaCast(const Tensor input, Tensor out) : Cast{input, out} {
    size_t shape_size = ndim_ * sizeof(*d_input_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);
    const size_t metadata_size = 2 * (shape_size + strides_size);
    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

    size_t offset = 0;
    d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_shape_.data(), shape_size);
    offset += shape_size;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_shape_.data(), shape_size);
    offset += shape_size;

    d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_strides_.data(), strides_size);
    offset += strides_size;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_strides_.data(), strides_size);

    Backend::Memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::MemcpyHostToDevice);
  }

  ~CudaCast() { Backend::Free(d_metadata_); }

  void operator()(const Tensor input, Tensor out) const override {
    int block_size = RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize();

    DispatchFunc<AllTypes, AllTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(input_dtype_), static_cast<int64_t>(out_dtype_),
         block_size},
        [&](auto list_tag) {
          using InT = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          using OutT = TypeMapType<Backend::kDeviceType, ListGet<1>(list_tag)>;
          constexpr int kBlockSize = ListGet<2>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));

          CastKernel<Backend::kDeviceType, InT, OutT, kBlockSize>
              <<<gridDims, blockDims, 0, cuda_stream>>>(
                  reinterpret_cast<OutT*>(out.data()),
                  reinterpret_cast<const InT*>(input.data()), d_out_shape_,
                  d_input_shape_, d_out_strides_, d_input_strides_,
                  output_size_, ndim_, is_out_contiguous_,
                  is_input_contiguous_);
        },
        "CudaCast::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
