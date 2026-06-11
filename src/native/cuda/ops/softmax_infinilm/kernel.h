#ifndef INFINI_OPS_CUDA_SOFTMAX_INFINILM_KERNEL_H_
#define INFINI_OPS_CUDA_SOFTMAX_INFINILM_KERNEL_H_

#include <algorithm>
#include <cstddef>
#include <cstring>
#include <vector>

#include "base/softmax_infinilm.h"
#include "common/generic_utils.h"
#include "data_type.h"
#include "dispatcher.h"
#include "native/cuda/kernel_commons.cuh"
#include "native/cuda/ops/softmax_infinilm/kernel.cuh"
#include "native/cuda/runtime_utils.h"

namespace infini::ops {

template <typename Backend>
class CudaSoftmaxInfinilm : public SoftmaxInfinilm {
 public:
  CudaSoftmaxInfinilm(const Tensor input, const int64_t dim,
                      const std::optional<DataType> dtype, Tensor out)
      : SoftmaxInfinilm{input, dim, dtype, out} {
    size_t shape_size = ndim_ * sizeof(*d_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);
    const size_t metadata_size = shape_size + 2 * strides_size;
    std::vector<std::byte> metadata(metadata_size);

    Backend::Malloc((void**)&d_metadata_, metadata_size);

    size_t offset = 0;
    d_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
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

  ~CudaSoftmaxInfinilm() { Backend::Free(d_metadata_); }

  void operator()(const Tensor input, const int64_t dim,
                  const std::optional<DataType> dtype,
                  Tensor out) const override {
    (void)dim;
    (void)dtype;
    auto cuda_stream =
        static_cast<typename Backend::Stream>(stream_ ? stream_ : 0);
    int block_size = std::min(
        RuntimeUtils<Backend::kDeviceType>::GetOptimalBlockSize(), 1024);

    DispatchFunc<AllFloatTypes, List<128, 256, 512, 1024>>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          const unsigned grid_x =
              static_cast<unsigned>(std::min<Tensor::Size>(row_count_, 65535));
          const unsigned grid_y = static_cast<unsigned>(
              utils::CeilDiv(row_count_, static_cast<Tensor::Size>(grid_x)));

          SoftmaxInfinilmKernel<kBlockSize, Backend::kDeviceType, T>
              <<<dim3(grid_x, grid_y), kBlockSize, 0, cuda_stream>>>(
                  reinterpret_cast<T*>(out.data()),
                  reinterpret_cast<const T*>(input.data()), d_shape_,
                  d_out_strides_, d_input_strides_, row_count_, dim_size_,
                  ndim_, dim_);
        },
        "CudaSoftmaxInfinilm::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
