#ifndef INFINI_OPS_CUDA_ADD_KERNEL_H_
#define INFINI_OPS_CUDA_ADD_KERNEL_H_

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <vector>

#include "base/add.h"
#include "common/generic_utils.h"
#include "cuda/add/kernel.cuh"
#include "cuda/kernel_commons.cuh"

namespace infini::ops {

template <typename Backend>
class CudaAdd : public Add {
 public:
  CudaAdd(const Tensor input, const Tensor other, Tensor out)
      : Add{input, other, out} {
    size_t shape_size = ndim_ * sizeof(*d_input_shape_);
    size_t strides_size = ndim_ * sizeof(*d_input_strides_);
    const size_t metadata_size = 3 * (shape_size + strides_size);
    std::vector<std::byte> metadata(metadata_size);

    Backend::malloc((void**)&d_metadata_, metadata_size);

    size_t offset = 0;
    d_input_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_shape_.data(), shape_size);
    offset += shape_size;

    d_other_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, other_shape_.data(), shape_size);
    offset += shape_size;

    d_out_shape_ = reinterpret_cast<Tensor::Size*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_shape_.data(), shape_size);
    offset += shape_size;

    d_input_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, input_strides_.data(), strides_size);
    offset += strides_size;

    d_other_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, other_strides_.data(), strides_size);
    offset += strides_size;

    d_out_strides_ = reinterpret_cast<Tensor::Stride*>(d_metadata_ + offset);
    std::memcpy(metadata.data() + offset, out_strides_.data(), strides_size);

    Backend::memcpy(d_metadata_, metadata.data(), metadata_size,
                    Backend::memcpyH2D);
  }

  ~CudaAdd() { Backend::free(d_metadata_); }

  void operator()(const Tensor input, const Tensor other,
                  Tensor out) const override {
    int block_size = Backend::GetOptimalBlockSize();
    DispatchFunc<AllTypes, AllCudaBlockSizes>(
        {static_cast<int64_t>(out_type_), block_size},
        [&](auto list_tag) {
          using T = TypeMapType<Backend::kDeviceType, ListGet<0>(list_tag)>;
          constexpr int kBlockSize = ListGet<1>(list_tag);

          auto cuda_stream =
              static_cast<typename Backend::stream_t>(stream_ ? stream_ : 0);
          dim3 blockDims(
              std::min(static_cast<Tensor::Size>(block_size), output_size_));
          dim3 gridDims(utils::CeilDiv(output_size_, blockDims.x));

          T* d_out = reinterpret_cast<T*>(out.data());
          const T* d_input = reinterpret_cast<const T*>(input.data());
          const T* d_other = reinterpret_cast<const T*>(other.data());

          AddKernel<Backend::kDeviceType, T, kBlockSize>
              <<<gridDims, blockDims, 0, cuda_stream>>>(
                  d_out, d_input, d_other, d_out_shape_, d_input_shape_,
                  d_other_shape_, d_out_strides_, d_input_strides_,
                  d_other_strides_, output_size_, ndim_, is_out_contiguous_,
                  is_input_contiguous_, is_other_contiguous_);
        },
        "CudaAdd::operator()");
  }

 private:
  std::byte* d_metadata_{nullptr};

  Tensor::Size* d_input_shape_{nullptr};

  Tensor::Size* d_other_shape_{nullptr};

  Tensor::Size* d_out_shape_{nullptr};

  Tensor::Stride* d_input_strides_{nullptr};

  Tensor::Stride* d_other_strides_{nullptr};

  Tensor::Stride* d_out_strides_{nullptr};
};

}  // namespace infini::ops

#endif
